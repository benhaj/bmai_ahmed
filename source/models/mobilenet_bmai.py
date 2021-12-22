#!/usr/bin/env python
# coding: utf-8

# In[2]:


from torch import nn
import torchvision



class Mobilenet_bmai:
    
    def __init__(self, img_size, SEXE=False, AGE=False, method_sex_age = 0,use_midas = False):
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
        model = torchvision.models.mobilenet_v2(pretrained=True)
        self.img_size = img_size
        self.SEXE = SEXE
        self.AGE = AGE
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
            
        if use_midas:
            first_layer= nn.Sequential(
                torch.nn.Conv2d(4,3,kernel_size=3,stride=1,padding=1,bias=True),
                torch.nn.ReLU()
            )
            model.features = torch.nn.Sequential(
                first_conv,
                model.features
            )
            
        if method_sex_age == 1 :
            # Instantiate new fully connected layer
            model.classifer = classifier_method_1(self.img_size)
            model.last = last_layer_method_1(self.AGE,self.SEXE)
        
        if method_sex_age == 2 :
            # Instantiate new fully connected layer
            model.classifer = classifier_method_2(self.img_size)
            model.last = last_layer_method_2(self.AGE,self.SEXE)
        
        if method_sex_age == 3 :
            # Instantiate new fully connected layer
            model.classifer = classifier_method_3(self.img_size)
            model.last = last_layer_method_3(self.AGE,self.SEXE)
        
        if method_sex_age == 4 :
            model.classifier = classifier_method_3(self.img_size)
            model.last = last_layer_method_3(False,False)
            
            model.mean_prediction = nn.Sequential(
                nn.Linear(in_features=2,out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features= 2)
            )
        model.name = 'mobilenet_v2'
        self.model = model
                

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

