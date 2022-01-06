#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import argparse
from prepare_dataset import *
from source.models.mobilenet_bmai import *
from trainer_bmai_2 import *
import torchvision
import torch
from source.models.OpenPose_bmai import *


# In[ ]:


img_size = 384
model_name = 'mobilenet'
SEXE=True
AGE=True
lr=0.005
SEED=0
method_sex_age=0


# In[2]:


model = torch.load('results/best_model_mobilenet_v2_cambodge.pt')
transforms = prepare_transforms()
dataset_guinee = bmaiDataset(csv_file=['/hdd/data/bmai_clean/full_guinee_data.csv'],img_size=img_size,transform=transforms)


# In[7]:


import numpy as np
import torch
import os
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim
from torch import nn
import pandas as pd
import wandb

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = model.to(device)

# Split data (train/test)
train_size = 0.8

num_train_entries = int(train_size * len(dataset_guinee))
num_test_entries = len(dataset_guinee) - num_train_entries
train_dataset, test_dataset = torch.utils.data.random_split(dataset_guinee,[num_train_entries,num_test_entries],generator=torch.Generator().manual_seed(SEED))

# Data loaders :
batch_size = 64
num_workers = 16

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

# Loss function and stored losses
losses = []
batch_losses = []

def loss_fn(y_pred,y_true):
    diff = torch.abs(y_pred-y_true)
    return torch.where(diff < (0.05*y_true),torch.zeros(1, 2,dtype=float).to(device),diff)
    
def test():
    model.eval()
    batch_losses = []
    y_true = []
    predictions = []
    predictions_branch = []
    for batch_idx, data in enumerate(test_dataloader):


         ## data in form ['img',sexe','days','height','weight']

        imgs = data[0].to(device)
        target = data[1:][0]
        num_elems_in_batch = target.shape[0]                ## Forward
        imgs = data[0].to(device)
        target = data[1:][0]
        num_elems_in_batch = target.shape[0]


        ## sex and age :
        sexe = target[:,0].reshape((num_elems_in_batch,1)).to(device)
        age = target[:,1].reshape((num_elems_in_batch,1)).to(device)

        ## Target:
        target = target[:,2:].to(device)

        ## Forward
        scores,mean_h_w = model(imgs,age,sexe)


        y_true.append(target.detach().numpy() if device=='cpu' else target.cpu().detach().numpy())                
        predictions.append(scores.detach().numpy() if device=='cpu' else scores.cpu().detach().numpy())
        predictions_branch.append(mean_h_w.cpu().detach().numpy())

        # loss
        loss = loss_fn(scores,target).sum()
        batch_losses.append(loss.item() if device=='cpu' else loss.cpu().item())


    average_loss = np.mean(batch_losses)
    print(f'Average test loss is {average_loss}')

    y_true= np.vstack(y_true)
    predictions = np.vstack(predictions)
    predictions_branch = np.vstack(predictions_branch) #### JUST TO SEE BRANCH PREDICTIONS

    mean_height_rel_error,mean_weight_rel_error = calculate_mean_absolute_error_results(y_true,predictions)
    print(f'mean_height_rel_error = {mean_height_rel_error}')
    print(f'mean_weight_rel_error = {mean_weight_rel_error}')

#     wandb.log({'epoch':epoch_num,'epoch_test_loss':average_loss, 'mean_height_rel_error':mean_height_rel_error, 'mean_weight_rel_error':mean_weight_rel_error})


    torch.save(y_true,'results/y_true_cambodge_on_guinee_mobilenet_v2_with_branch.pt')
    torch.save(predictions,f'results/predictions_cambodge_on_guinee_mobilenet_v2_with_branch.pt')
    torch.save(predictions_branch,f'results/branch_predictions_cambodge_on_guinee_mobilenet_v2_with_branch.pt')


    return mean_height_rel_error,mean_weight_rel_error#,average_loss




def calculate_mean_absolute_error_results(y_true,predictions):
    df = pd.DataFrame()
    df['true_height'] = y_true[:,0]
    df['true_weight'] = y_true[:,1]
    df['predicted_height'] = predictions[:,0]
    df['predicted_weight'] = predictions[:,1]

    df['height_rel_err'] = df.apply(lambda row : np.abs(row.true_height - row.predicted_height)/row.true_height,axis=1)
    df['weight_rel_err'] = df.apply(lambda row : np.abs(row.true_weight - row.predicted_weight)/row.true_weight,axis=1)

    mean_height_rel_error = df.height_rel_err.values.mean()
    mean_weight_rel_error = df.weight_rel_err.values.mean()

    return mean_height_rel_error,mean_weight_rel_error



# In[ ]:

test()


