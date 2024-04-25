#!/usr/bin/env python
import argparse

import numpy as np
import torch

from .io import IO
import torchlight
import tools.utils as utils
from torchlight import DictAction
from torchlight import import_class

import cv2

class DemoSkeleton(IO):
    """ A demo for reading in pre-processed Kinetics Skeleton data in
    a NPY format, and visualising it.

    The purpose of this demo is to ensure the output NPY data is at the very
    least not ineligible; it is apparant that if the skeleton data is recognizable
    as human, then `fd_gendata.py` is working correctly, besides some very small
    potential rounding issues.
    """

    # Compared to other demos, we need to load data so the feeder has to be imported.
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        self.data_loader['skeleton'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=1,
            shuffle=False,
            num_workers=self.arg.num_worker * torchlight.ngpu(
                self.arg.device))

    def start(self):

        # load required skeleton data from npy file
        self.load_data()
        loader = self.data_loader['skeleton']
        loader = iter(loader)

        # start skeleton processing
        while(True):

            #get image
            orig_image = np.zeros((200,200,3))
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape
            
            
            # get data
            (feddata, category) = next(loader)   
            feddata = feddata.float().to("cuda:0")
            category = category.float().to("cuda:0")
            
            # if torch.equal(category, torch.tensor([258.], device='cuda:0')):
            print("Start:")
            feddata = feddata[0]
            for i in range (feddata.shape[1]):
                sub_array = feddata[:, i:i+1, :, :]
                #getting each frame; render each frame here
                image = self.render(sub_array , orig_image, 1)
                cv2.imshow("ST-GCN", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    def render(self, data_numpy, orig_image, fps=0):
        images = utils.visualization.stgcn_visualize_skeleton(
            data_numpy[:, [-1]],
            self.model.graph.edge,
            [orig_image],
            self.arg.height,
            fps=fps)
        image = next(images)
        image = image.astype(np.uint8)
        return image
 
    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Skeleton Demo for Spatial Temporal Graph Convolution Network')

        # feeder args
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')

        # region arguments yapf: disable
        parser.add_argument('--video',
                            default='./resource/media/ta_chi.mp4',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--model_input_frame',
                            default=128,
                            type=int)
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--height',
                            default=800,
                            type=int,
                            help='height of frame in the output video.')
        parser.set_defaults(
            config='./config/st_gcn/kinetics-skeleton/demo_skeleton.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser