#!/usr/bin/env python
# coding: utf-8

# In[2]:

import torch
from torch import nn
import torchvision



class Mobilenet_bmai(nn.Module):
    
    def __init__(self, img_size, SEXE=False, AGE=False, method_sex_age = 0):
        """
        Initializes the modified mobilnet model depending on how we want to add sex and age attributes
        :param SEX : Boolean, if we use sex attribute or not.
        :param AGE : Boolean, if we use age attribute or not.
        :param method_sex_age: This parameter can take 4 different values:
                    0 (default) : Do not add sex and age attribute to the model.
                    1 : Add attributes (SEX/AGE) that are set to True, as input to the first Fully connected layer
                    2 : Add attributes (SEX/AGE) that are set to True, as input to the second Fully connected layer
                    3 : Add attributes (SEX/AGE) that are set to True, as input to the last Fully connected layer
                    4 : Add attributes (SEX/AGE) that are set to True, as input to the another model that will be used
                to predict the mean_height and mean_weight. In this case, our Mobilenet model will predict the difference
                between the True H/W and the Mean H/W. 
        """
        super().__init__()
        self.features = torchvision.models.mobilenet_v2(pretrained=True).features
        self.img_size = img_size
        self.SEXE = SEXE
        self.AGE = AGE
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        if method_sex_age == 0 :
            # Instantiate new fully connected layer
            self.classifier = classifier_method_1(self.img_size)
            self.last = last_layer_method_1(self.AGE,self.SEXE)
        
        if method_sex_age == 1 :
            # Instantiate new fully connected layer
            self.classifier = classifier_method_1(self.img_size)
            self.last = last_layer_method_1(self.AGE,self.SEXE)
        
        if method_sex_age == 2 :
            # Instantiate new fully connected layer
            self.classifier = classifier_method_2(self.img_size)
            self.last = last_layer_method_2(self.AGE,self.SEXE)
        
        if method_sex_age == 3 :
            # Instantiate new fully connected layer
            self.classifier = classifier_method_3(self.img_size)
            self.last = last_layer_method_3(self.AGE,self.SEXE)
        
        if method_sex_age == 4 :
            self.classifier = classifier_method_3(self.img_size)
            self.last = last_layer_method_3(False,False)
            
            self.mean_prediction = nn.Sequential(
                nn.Linear(in_features=2,out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features= 2)
            )
        self.name = 'mobilenet_v2'
        #self.model = model
        self.method_sex_age = method_sex_age
    
    
    def forward(self,imgs,age,sexe):
        if self.AGE & self.SEXE:
            if self.method_sex_age == 4 :
                scores = self.last(self.classifier(self.features(imgs)))
                mean_h_w = self.mean_prediction(torch.cat([age.float(),sexe.float()],dim=1))
                scores = torch.add(scores, mean_h_w)
                return scores,mean_h_w
            else:
                feat = self.classifier(self.features(imgs))
                concat = torch.cat([feat,sexe,age],dim=1).float()
                scores = self.last(concat)

        elif self.AGE:
            feat = self.classifier(self.features(imgs))
            concat = torch.cat([feat,age],dim=1).float()
            scores = self.last(concat)

        elif self.SEXE:
            feat = self.classifier(self.features(imgs))
            concat = torch.cat([feat,sexe],dim=1).float()
            scores = self.last(concat)
        else:
            feat = self.features(imgs)
            classi = self.classifier(feat)
            scores = self.last(classi)
        
        return scores
           

def classifier_method_1(img_size):
    if img_size==256:
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280*8*8, out_features=256),
            nn.ReLU()
        )
    else:
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280*12*12, out_features=256),
            nn.ReLU()
        )
    return classifier


def last_layer_method_1(AGE,SEXE):
    if AGE and SEXE:
        last = nn.Sequential(
            nn.Linear(in_features=256+2,out_features=128), ## add two inputs
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )
    elif AGE or SEXE:
        last = nn.Sequential(
            nn.Linear(in_features=256+1,out_features=128), ## add only one input
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )
    else:
        last = nn.Sequential(
            nn.Linear(in_features=256+0,out_features=128), ## add no input
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )
    return last

def classifier_method_2(img_size):
    if img_size==256:
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280*8*8, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
        )
    else:
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280*12*12, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
        )
    return classifier


def last_layer_method_2(AGE,SEXE):
    if AGE and SEXE:
        last = nn.Sequential(
            nn.Linear(128+2,32), ## add two inputs
            nn.ReLU(),
            nn.Linear(32,2)
        )
    elif AGE or SEXE:
        last = nn.Sequential(
            nn.Linear(128+1,32),## add only one input
            nn.ReLU(),
            nn.Linear(32,2)
        )
    else:
        last = nn.Sequential(
            nn.Linear(128,32),  ## add no input
            nn.ReLU(),
            nn.Linear(32,2)
        )
    return last

def classifier_method_3(img_size):
    if img_size==256:
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280*8*8, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
        )
    else:
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280*12*12, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
        )
    return classifier


def last_layer_method_3(AGE,SEXE):
    if AGE and SEXE:
        last = nn.Sequential(
            nn.Linear(32+2,2) ## add two inputs
        )
    elif AGE or SEXE:
        last = nn.Sequential( 
            nn.Linear(32+1,2)  ## add one inputs
        )
    else:
        last = nn.Sequential(
            nn.Linear(32,2)
        )
    return last

