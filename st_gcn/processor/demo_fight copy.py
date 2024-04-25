#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time
import csv
from sklearn.metrics import f1_score

import numpy as np
import torch
import skvideo.io

from .io import IO
import tools
import tools.utils as utils

import cv2

class ImageState:
    def __init__(self):
        self.showSkeleton = False
        self.showRGB = True
        self.buttonSkeleton = [0,30,0,150]
        self.buttonRGB = [30,60,0,150]

    # function that handles the mousclicks
    def process_click(self,event, x, y,flags, params):
        # check if the click is within the dimensions of the button
        if event == cv2.EVENT_LBUTTONDOWN:
            if y > self.buttonSkeleton[0] and y < self.buttonSkeleton[1] and x > self.buttonSkeleton[2] and x < self.buttonSkeleton[3]:   
                self.showSkeleton = not self.showSkeleton
            if y > self.buttonRGB[0] and y < self.buttonRGB[1] and x > self.buttonRGB[2] and x < self.buttonRGB[3]:   
                self.showRGB = not self.showRGB

class DemoFight(IO):
    """ A demo for utilizing st-gcn in the realtime action recognition.
    The Openpose python-api is required for this demo.

    Since the pre-trained model is trained on videos with 30fps,
    and Openpose is hard to achieve this high speed in the single GPU,
    if you want to predict actions by **camera** in realtime,
    either data interpolation or new pre-trained model
    is required.

    Pull requests are always welcome.
    """

    def start(self):
        self.predictions = []
        self.totalpredictions = []
        self.groundtruth = []
        if(self.arg.mode == "test"): self.test()
        else: self.recognition()

    def test(self):
        with open(self.arg.test_split, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for video in csvreader:
                self.predictions = []
                self.videogroundtruth = []
                self.arg.video = self.arg.video_path+video[0]+'.mp4'
                annotation = self.arg.annotation_path+video[0]+'.csv'
                with open(annotation, newline='') as annotationFile:
                    for index in annotationFile: 
                        self.groundtruth.append(int(index[0]))
                        self.videogroundtruth.append(int(index[0]))
                self.recognition()
                with open(self.arg.prediction_out+video[0]+'.csv', 'w', newline='') as predictionsfile:
                    wr = csv.writer(predictionsfile, quoting=csv.QUOTE_ALL)
                    wr.writerow(self.predictions)
                self.totalpredictions.extend(self.predictions)
                with open(self.arg.prediction_out+video[0]+'-f1.csv', 'w', newline='') as localf1file:
                    f1 = f1_score(self.predictions, self.videogroundtruth, average='macro')
                    wr = csv.writer(localf1file, quoting=csv.QUOTE_ALL)
                    wr.writerow(self.predictions)
                
        f1 = f1_score(self.totalpredictions, self.groundtruth, average='macro')
        print(self.totalpredictions)
        print(self.groundtruth)
        print('\tF1 Score: {:.4f}'.format(f1))


    def recognition(self):
        imageState = ImageState()

        # load openpose python api
        if self.arg.openpose is not None:
            sys.path.append('{}/python'.format(self.arg.openpose))
            sys.path.append('{}/build/python'.format(self.arg.openpose))
        try:
            import pyopenpose as op
        except:
            print('Can not find Openpose Python API.')
            return

        video_name = self.arg.video.split('/')[-1].split('.')[0]
        label_name_path = './resource/fd/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # initiate
        opWrapper = op.WrapperPython()
        params = dict(model_folder='./models', model_pose='COCO',output_resolution="480x270")
        opWrapper.configure(params)
        opWrapper.start()
        self.model.eval()
        pose_tracker = naive_pose_tracker()

        if self.arg.video == 'camera_source':
            video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            video_capture = cv2.VideoCapture(self.arg.video)

        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,100)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,100)

        # start recognition
        start_time = time.time()
        frame_index = 0
        app_fps = 1
        tic = 0
        voting_label_name = "no fighting"
        while(True):

            tic = time.time()

            # get image
            ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape
            
            # pose estimation
            datum = op.Datum()
            datum.cvInputData = orig_image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3)
            if multi_pose is not None:
                if len(multi_pose.shape) != 3:
                    continue

                # normalization
                multi_pose[:, :, 0] = multi_pose[:, :, 0]/W
                multi_pose[:, :, 1] = multi_pose[:, :, 1]/H
                multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
                multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
                multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

                # pose tracking
                if self.arg.video == 'camera_source':
                    frame_index = int((time.time() - start_time)*self.arg.model_fps)
                else:
                    frame_index += 1
                pose_tracker.update(multi_pose, frame_index)
                data_numpy = pose_tracker.get_skeleton_sequence()
                data = torch.from_numpy(data_numpy)
                data = data.unsqueeze(0)
                data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

                # model predict
                voting_label_name, video_label_name, output, intensity = self.predict(
                    data)

                # visualization
                app_fps = 1 / (time.time() - tic)
                image = self.render(imageState.showSkeleton, imageState.showRGB, data_numpy, voting_label_name,
                                    video_label_name, intensity, orig_image, app_fps)
            else:
                tic = time.time() - (1 / (app_fps))
                app_fps = 1 / (time.time() - tic)
                image = self.renderEmpty(imageState.showSkeleton, imageState.showRGB, orig_image, voting_label_name, app_fps)

            if(voting_label_name == 'no fighting'): 
                self.predictions.append(0)
            else: 
                self.predictions.append(1)

            control_image = np.zeros((imageState.buttonRGB[1],imageState.buttonRGB[3]), np.uint8)
            control_image[imageState.buttonSkeleton[0]:imageState.buttonRGB[1],imageState.buttonSkeleton[2]:imageState.buttonRGB[3]] = 320

            if(imageState.showSkeleton):cv2.putText(control_image, 'Disable Skeleton',(0,20),cv2.FONT_HERSHEY_COMPLEX, 0.5,0)
            else:cv2.putText(control_image, 'Enable Skeleton',(0,20),cv2.FONT_HERSHEY_COMPLEX, 0.5,0)

            if(imageState.showRGB):cv2.putText(control_image, 'Disable RGB',(0,50),cv2.FONT_HERSHEY_COMPLEX, 0.5,0)
            else:cv2.putText(control_image, 'Enable RGB',(0,50),cv2.FONT_HERSHEY_COMPLEX, 0.5,0)


            image[0:control_image.shape[0], 0:control_image.shape[1],0] = control_image[0:control_image.shape[0], 0:control_image.shape[1]]
            image[0:control_image.shape[0], 0:control_image.shape[1],1] = control_image[0:control_image.shape[0], 0:control_image.shape[1]]
            image[0:control_image.shape[0], 0:control_image.shape[1],2] = control_image[0:control_image.shape[0], 0:control_image.shape[1]]

            cv2.namedWindow('Control')
            cv2.setMouseCallback('Control',imageState.process_click)

            cv2.imshow("Control", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def predict(self, data):
        # forward
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l]
                                   for l in latest_frame_label]

        print(output.shape)
        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        print(voting_label_name)
        for t in range(num_frame):
            frame_label_name = list()
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)
        return voting_label_name, video_label_name, output, intensity

    def render(self, showSkeleton, showRGB, data_numpy, voting_label_name, video_label_name, intensity, orig_image, fps=0):
        images = utils.visualization.stgcn_fight_visualize(
            showSkeleton,
            showRGB,
            data_numpy[:, [-1]],
            self.model.graph.edge,
            intensity[[-1]], [orig_image],
            voting_label_name,
            [video_label_name[-1]],
            self.arg.height,
            fps=fps)
        image = next(images)
        image = image.astype(np.uint8)
        return image

    def renderEmpty(self, showSkeleton, showRGB, orig_image, voting_label_name , fps=0):
        images = utils.visualization.stgcn_visualize_fight_empty(
            showSkeleton,
            showRGB,
            [orig_image],
            voting_label_name,
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
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--video',
                            default='./resource/media/ta_chi.mp4',
                            help='Path to video')
        parser.add_argument('--mode',
                            default='recognition',
                            help='Execution mode of demo; test or recognise')
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
        parser.add_argument('--annotation_path',
                            default='',
                            help='Path to temporal-level annotations for test set')
        parser.add_argument('--video_path',
                            default='',
                            help='Path to all videos in test dataset')
        parser.add_argument('--test_split',
                            default='',
                            help='Path to name of videos for test set')
        parser.add_argument('--prediction_out',
                            default='',
                            help='Path to output of predictions')
        parser.set_defaults(
            config='./config/fd/demo_fight.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser

class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame-latest_frame-1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close
