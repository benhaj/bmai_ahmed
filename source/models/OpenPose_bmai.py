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




### MODEL 2

class OpenPose_first_and_second_Part_of_InitialStage(nn.Module):
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
            conv_dw(512, 512)   # conv5_5
         )
        # self.cpm = Cpm(256, num_channels)
        # self.initial_stage = InitialStage(num_channels)
    
    
    def forward(self, x):
        backbone_features = self.model(x)
        # backbone_features = self.cpm(backbone_features)
        # stages_output = self.initial_stage(backbone_features)
        return backbone_features

class Baseline_2(nn.Module):

    def __init__(self,inference_model,freeze=False,SEXE=False,AGE=False,OUTPUT_SIZE=(32,57)):
        super().__init__()
        self.num_channels = inference_model.num_channels
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.inference_model = inference_model
        self.SEXE = SEXE
        self.AGE = AGE
        if freeze:
            for param in inference_model.parameters():
                param.requires_grad = False
        self.base = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1),
            nn.AdaptiveAvgPool2d(output_size=OUTPUT_SIZE)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(in_features=OUTPUT_SIZE[0]*OUTPUT_SIZE[1],out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU()
            
        )
        
        if AGE & SEXE:
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
        base_output = self.base(inference_output)
        classifier_output = self.classifier(base_output)
        if self.SEXE & self.AGE:
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




def prepare_OpenPose_model(freeze=True):
    model = OpenPose_first_and_second_Part_of_InitialStage()
    load_state(model, checkpoint)
    model = Baseline_2(model,freeze=True)
    return model

