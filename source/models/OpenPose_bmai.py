#!/usr/bin/env python
# coding: utf-8



import torch
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import math
import cv2
import torchvision
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import collections


# In[ ]:


from source.modules.conv import *

## MODEL 1 (full lightweight openpose model, with PAFs output summed.

class Baseline_1(nn.Module):

    def __init__(self,inference_model,freeze=False,IMG_SIZE=256):
        super().__init__()
        self.name = "baseline1_freeze" if freeze else "baseline1_NoFreeze"
        self.num_channels = inference_model.num_channels
        self.in_features = int((IMG_SIZE/8) * (IMG_SIZE/8))
        self.inference_model = inference_model
        
        if freeze:
            for param in inference_model.parameters():
                param.requires_grad = False

        self.last = nn.Sequential(
            conv_dw(512, 64),
            conv_dw(64, 1),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(in_features=self.in_features,out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=2)
        )    

    def forward(self, x):
        inference_output = self.inference_model(x)
        last_out = self.last(inference_output)
        return last_out


### MODEL 2

class Mobilenet_v1_part_of_OpenPose(nn.Module):
    def __init__(self, num_channels=128):
        super().__init__()
        self.num_channels=num_channels
        self.model = nn.Sequential(
            conv(3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512), # conv5_5
         )
    
    def forward(self, x):
        backbone_features = self.model(x)
        return backbone_features

class Baseline_2(nn.Module):

    def __init__(self,inference_model,freeze=False,SEXE=False,AGE=False,method_age_sex=4,OUTPUT_SIZE=(32,57)):
        super().__init__()
        self.num_channels = inference_model.num_channels
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.inference_model = inference_model
        self.SEXE = SEXE
        self.AGE = AGE
        self.method_age_sex = method_age_sex

        if freeze:
            for param in inference_model.parameters():
                param.requires_grad = False
        self.base = nn.Sequential(
            #nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1),
            #nn.AdaptiveAvgPool2d(output_size=OUTPUT_SIZE)
        )

        self.classifier = nn.Sequential(
            #nn.Dropout(0.1),
            nn.Flatten(),
            #nn.Linear(in_features=OUTPUT_SIZE[0]*OUTPUT_SIZE[1],out_features=256),
            nn.Linear(in_features=48*48,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU()
            
        )
        
        if AGE & SEXE:
            if self.method_age_sex==4:
                model.mean_prediction = nn.Sequential(
                    nn.Linear(in_features=2,out_features=32),
                    nn.ReLU(),
                    nn.Linear(in_features=32, out_features=64),
                    nn.ReLU(),
                    nn.Linear(in_features=64, out_features=32),
                    nn.ReLU(),
                    nn.Linear(in_features=32, out_features= 2)
                )
                self.last = nn.Sequential(
                    # nn.Dropout(0.2),
                    nn.Linear(in_features=128,out_features=32),
                    nn.ReLU(),
                    nn.Linear(in_features=32,out_features=2)
                )
            else:
                self.last = nn.Sequential(
                    # nn.Dropout(0.2),
                    nn.Linear(in_features=128+2,out_features=32),
                    nn.ReLU(),
                    nn.Linear(in_features=32,out_features=2)
                )

        elif AGE or SEXE :
            self.last = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(in_features=128+1,out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32,out_features=2)
            )
        else:
            self.last = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(in_features=128,out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32,out_features=2)
            )

    def forward(self, x,age,sexe):
        inference_output = self.inference_model(x)
        classifier_output = self.classifier(inference_output)
        if self.SEXE & self.AGE:
            if self.method_age_sex==4:
                mean_h_w = self.mean_prediction(age,sexe)
                diff_to_mean = self.last(classifier_output)
                last_out = torch.add(diff_to_mean, mean_h_w)
                return last_out
            else:
                concat = torch.cat([classifier_output,age,sexe],dim=1).float()
        elif self.SEXE:
            concat = torch.cat([classifier_output,sexe],dim=1).float()
        elif self.AGE:
            concat = torch.cat([classifier_output,age],dim=1).float()
        else:
            concat = classifier_output

        last_out = self.last(concat)
        return last_out


def load_state(net):
    checkpoint = torch.load("checkpoint_iter_370000.pth", map_location='cpu')
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)




# def prepare_OpenPose_model(freeze=True,method_age_sex=0):
#     model = OpenPose_first_and_second_Part_of_InitialStage()
#     load_state(model)
#     model = Baseline_2(model,freeze=True,method_age_sex=method_age_sex)
#     model.name = 'OpenPose_bmai'
#     return model



"""
Taken as is from: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
"""
from source.modules.conv import conv, conv_dw, conv_dw_no_bn
from torch import nn
import torch


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
#         return [heatmaps, pafs]
        print(pafs.shape)
        return np.sum(pafs,axis=2)

class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]
#         return torch.sum(pafs,dim=1)


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.num_channels=num_channels

        self.model = nn.Sequential(
            conv(3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))
        ## select pafs, arrange channels and sum them
        pafs = stages_output[-1]
        pafs = transforms.Resize((192, 192), interpolation=transforms.InterpolationMode.BICUBIC)(pafs)
        return torch.sum(pafs,dim=1)


def prepare_OpenPose_model(freeze=True,method_age_sex=0):
    model = Mobilenet_v1_part_of_OpenPose()
    load_state(model)
    model = Baseline_1(model,freeze=True)
    model.name = 'OpenPose_bmai'
    return model
