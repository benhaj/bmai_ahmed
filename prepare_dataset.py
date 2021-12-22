#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from skimage import io , transform
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import torch
import numpy as np
import cv2
import math


HEIGHT_SIZE = 256
IMG_MEAN = np.array([0.5, 0.5, 0.5], np.float32)


class bmaiDataset(Dataset):
    """bmai dataset.
    This class instantiate a bmai dataset.
    
    
    If we want to use a csv file and perform transform on each image/batch specify :
        use_csv : Set use_csv = True (!!)
        csv_file : path to our csv_file
        transfrom : transform to be applied (prepare_transforms() will return a composed element (Normalization+Resize+Pad, see below)
    
    In case you want to directly upload transformed images and their labels specify:
        images : already transformed images
        labels : their corresponding labels
        transform : transforms.ToTensor() in case transformed_images are in PIL format (my case)
                    otherwise you can add any transform you want, or None
                    
 
    """
    def __init__(self, csv_file = None , img_size = 256 , transform =None, use_midas = False):
        
        ## annotations in form ['img',sexe','days','height','weight']
        self.use_midas=use_midas
        if self.use_midas:
            model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
            #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
            #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
            midas = torch.hub.load("intel-isl/MiDaS", model_type)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            midas.to(device)
            midas.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                midas_transform = midas_transforms.dpt_transform
            else:
                midas_transform = midas_transforms.small_transform
        
        if len(csv_file)==1:
            self.annotations = pd.read_csv(csv_file[0])
        else: ## len(csv_file)==2 (guinee + cambodge)
            annotations_1 = pd.read_csv(csv_file[0])
            annotations_2 = pd.read_csv(csv_file[1])
            self.annotations = pd.concat([annotations_1,annotations_2],ignore_index=True)
            
        ## to normalize days/age
        self.age_mean = self.annotations.iloc[:,2].values.mean()
        self.age_std = self.annotations.iloc[:,2].values.std()
                    
        self.transform = transform
        self.img_size=img_size

        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ## ['img',sexe','days','height','weight']
        
        img_path = self.annotations.iloc[index, 0]#.values[0]
        img_path = "/hdd/data/" + img_path.replace('data','bmai_clean',1)
        
        new_path = prepare_new_path(img_path,self.img_size)
        img = io.imread(new_path)
        
        if self.use_midas:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_batch = transform(img).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()
            
        sexe = self.annotations.iloc[index, 1]
        days = (self.annotations.iloc[index,2] - self.age_mean) / self.age_std
        height_weight = self.annotations.iloc[index,3:].values.astype(float)
        y_label = torch.tensor(np.hstack([sexe,days,height_weight]))
        
        if use_midas:
            concatenated_img = np.empty(shape=(img.shape[0],img.shape[1],img.shape[2]+1))
            concatenated_img[:,:,:3] = img.copy()
            concatenated_img[:,:,3] = output
            img = concatenated_img.copy()

        if self.transform:
            img=self.transform(img.copy())
            
        
        return img , y_label

### RESCALE (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        
        h, w, c = sample.shape[:]
        
        if h>=w:
            ratio = h/self.output_size
            new_h, new_w = round(h / ratio), round(w / ratio)
        else:
            ratio = w/self.output_size
            new_h, new_w = round(h / ratio), round(w / ratio)
        
        img = transform.resize(sample, (new_h, self.output_size, c))
        out = torch.from_numpy(img).permute(2,0,1)
        return out


## Helper function for padding
def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img



class Pad(object):
    
    def __init__(self,stride,pad_value):
        self.stride = stride
        self.pad_value = pad_value
    
    def __call__(self, sample):
        img = sample.permute(1,2,0).numpy()
        min_dims = [HEIGHT_SIZE, max(img.shape[1], HEIGHT_SIZE)]
        padded_img = pad_width(img, self.stride, self.pad_value,min_dims)
        return torch.from_numpy(padded_img).permute(2, 0, 1).float()


    
def prepare_transforms(img_mean=IMG_MEAN,img_std=1):
    normalization = transforms.Normalize(IMG_MEAN,1)
    composed = transforms.Compose([transforms.ToTensor(),
                                   normalization])
    #                               transforms.Resize((HEIGHT_SIZE,HEIGHT_SIZE)),
    #                           Pad(stride=8, pad_value=(0, 0, 0))
    #                          ])
    return composed


def prepare_new_path(path,size):
    splitted = path.split('.')
    if len(splitted)!=2:
        new_path = f'{splitted[0]}.{splitted[1]}_{size}.{splitted[2]}'
    else:
        new_path = f'{splitted[0]}_{size}.{splitted[1]}'
    return new_path


def create_df_entry(args,best):
    entry = {
        'data':args.data_name,
        'img_size':args.img_size,
        'sexe':args.SEXE,
        'age':args.AGE,
        'seed':args.SEED,
        'epochs':args.epochs,
        'lr':args.lr,
        'height_rel_err':best[0],
        'weight_rel_err':best[1]
    }
    return entry


def prepare_dataset(data_name,img_size):
    ## Data
    # TRANSFORMS :
    transforms = prepare_transforms()
    if data_name == 'guinee':
        dataset = bmaiDataset(csv_file=['/hdd/data/bmai_clean/full_guinee_data.csv'],img_size=img_size,transform=transforms)
    elif data_name == 'cambodge':
        dataset = bmaiDataset(csv_file=['/hdd/data/bmai_clean/full_cambodge_data.csv'],img_size=img_size,transform=transforms)
    else:
        dataset = bmaiDataset(csv_file=['/hdd/data/bmai_clean/full_guinee_data.csv','/hdd/data/bmai_clean/full_cambodge_data.csv'],img_size=img_size,transform=transforms)
    return dataset
