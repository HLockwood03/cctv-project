#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import tools.utils as utils
import cv2

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from net.st_gcn import st_gcn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Transfer_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """
    
    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        
    
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.cross_val != True:
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.5**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def transfer(self):

        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        self.io.print_log("\t\tweight decay " + str(self.optimizer.state_dict().get('param_groups')[0].get('weight_decay')) + " lr " + str(self.optimizer.state_dict().get('param_groups')[0].get('lr')))
        for data, label in loader:
            data, label = data.cuda(), label.cuda()
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.meta_info['iter'] += 1


        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()

    def val(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['val']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.prev_loss = self.current_loss
            self.current_loss = np.mean(loss_value)
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            f1 = f1_score(self.label, np.argmax(self.result, axis=1), average='macro')
            self.io.print_log('\tF1 Score: {:.4f}'.format(f1))
            self.prev_f1 = self.current_f1
            self.current_f1 = f1

    def render(self, data_numpy, orig_image, fps=0):
        images = utils.visualization.stgcn_visualize_skeleton(
            data_numpy[:, [-1]],
            self.model.graph.edge,
            [orig_image],
            500,
            fps=fps)
        image = next(images)
        image = image.astype(np.uint8)
        return image

    def start(self):
        Feeder = import_class(self.arg.feeder)
        self.running_f1 = 0
        if self.arg.cross_val:
            k_folds = 5
            kf = KFold(n_splits=k_folds, shuffle=True)
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(Feeder(**self.arg.train_feeder_args))):
                self.io.print_log(f"Fold {fold+1}")
                self.data_loader['train'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.train_full_feeder_args),
                    batch_size=self.arg.batch_size,
                    num_workers=self.arg.num_worker * torchlight.ngpu(
                        self.arg.device),
                    sampler=torch.utils.data.SubsetRandomSampler(train_idx))
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.train_full_feeder_args),
                    batch_size=self.arg.test_batch_size,
                    num_workers=self.arg.num_worker * torchlight.ngpu(
                        self.arg.device),
                    sampler=torch.utils.data.SubsetRandomSampler(test_idx))

                self.process()

            self.io.print_log(f"Average F1 {self.running_f1 / k_folds}")

        else:
            self.process()

    def process(self):
        dropout_vals = [0.1]#,0.25,0.1]
        weight_decay_vals = [0.0001]#, 0.0005, 0.001]
        learning_rate_vals = [0.015]#, 0.01]
        best_f1 = 0
        best_params = ""

        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        print(self.arg.model)
        initial_model = self.model
        self.last_model = self.model

        unique_classes = set()
        for _ , labels in self.data_loader['train']:
            unique_classes.update(labels.numpy().tolist())
        num_classes = len(unique_classes)

        # the path of weights must be appointed
        if self.arg.weights is None:
            raise ValueError('Please appoint --weights.')
        self.io.print_log('Model:   {}.'.format(self.arg.model))
        self.io.print_log('Weights: {}.'.format(self.arg.weights))

        for dropout in dropout_vals:
            for weight_decay in weight_decay_vals:
                for learning_rate in learning_rate_vals:
                    param_string = "["+str(dropout)+","+str(weight_decay)+","+str(learning_rate)+"]"
                    self.io.print_log("=========================="+param_string+"==========================")
                    self.arg.weight_decay = weight_decay
                    self.arg.base_lr = learning_rate
                    self.lr = learning_rate                    
                    
                    self.model = initial_model
                    self.load_optimizer()

                    for idx , layer in enumerate(self.model.st_gcn_networks):
                        layer.tcn[4] = nn.Dropout(dropout, inplace=True)
                        if idx < 10:
                            for param in layer.parameters():
                                param.requires_grad = False
                        else: break

                    for param in self.model.data_bn.parameters():
                        param.requires_grad = False
                    
                    self.model.fcn = nn.Conv2d(256, num_classes, kernel_size=1) # New model will have an output with the size being number of classes      
                    self.model.to("cuda")
            
                    summary(self.model, input_size=(16,3,300,18,2), verbose=1, depth=4, col_names=["input_size","output_size","num_params","trainable"],col_width=20, row_settings=["var_names"])

                    self.current_loss = 1
                    self.prev_loss = 1
                    self.current_f1 = 0   

                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        # for param in self.model.fcn.parameters():
                        #     print(param)
                        self.meta_info['epoch'] = epoch

                        # training
                        self.io.print_log('Transfer epoch: {}'.format(epoch))
                        self.transfer()

                        # test
                        self.io.print_log('Eval epoch: {}'.format(epoch))
                        self.val()
                        self.io.print_log('Done.')

                        # early stopping; if loss increases
                        if(self.prev_loss < self.current_loss):
                            print("Loss increased. Early stopping.")
                            break

                        self.last_model = self.model

                    # save model
                    self.io.save_model(self.last_model, 'model_1l.pt')

                    self.lr = 0.0005
                    self.arg.base_lr = 0.0005
                    self.model = self.last_model
                    for idx , layer in enumerate(self.model.st_gcn_networks):
                        if idx > 6:
                            for param in layer.parameters():
                                param.requires_grad = True

                    # save the output of model
                    if self.arg.save_result:
                        result_dict = dict(
                        zip(self.data_loader['val'].dataset.sample_name,
                            self.result))
                        self.io.save_pkl(result_dict, 'test_result.pkl')

                    self.model.to("cuda")
                    self.current_f1 = 0       
                    #summary(self.model, input_size=(16,3,300,18,2), verbose=1, col_names=["input_size","output_size","num_params","trainable"],col_width=20, row_settings=["var_names"])
                    
                    self.io.print_log("Fine Tuning start")
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        self.meta_info['epoch'] = epoch

                        # training
                        self.io.print_log('Transfer epoch: {}'.format(epoch))
                        self.transfer()

                        # test
                        self.io.print_log('Eval epoch: {}'.format(epoch))
                        self.val()
                        self.io.print_log('Done.')

                        # early stopping; if loss increases
                        if(self.prev_loss < self.current_loss):
                            print("Loss increased. Early stopping.")
                            self.current_f1 = self.prev_f1
                            break

                    # save model
                    self.io.save_model(self.model, 'model_1l_ft.pt'.format(epoch + 1))
                    if(self.current_f1 > best_f1):
                        best_f1 = self.current_f1
                        best_params = param_string

        self.io.print_log("BEST PARAMS ARE "+best_params)
        self.io.print_log("F1 WAS "+str(best_f1))
        self.running_f1 += best_f1
        # # evaluation
        # self.io.print_log('Evaluation Start:')
        # self.test()
        # self.io.print_log('Done.\n')

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network',
            conflict_handler='resolve')

        parser.add_argument('-c', '--config', default='./config/ucf/transfer.yaml', help='path to the configuration file')
        

        #default args
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--train_full_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training crossval')
        parser.add_argument('--cross_val', type=bool, default=False, help='the toggle for if doing cross validation or not')

        #transfer args
        parser.add_argument('--class_list', help='link to list of new classes')


        return parser
