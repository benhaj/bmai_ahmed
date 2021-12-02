#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
from skimage import io, transform
from torchvision import transforms
import numpy as np 
import torch
from prepare_dataset import Pad



HEIGHTS = [256 , 384]
IMG_MEAN = np.array([0.5, 0.5, 0.5], np.float32)
normalization = transforms.Normalize(IMG_MEAN,1)


# In[38]:


df_guinee = pd.read_csv('data/full_guinee_data.csv')
df_cambodge = pd.read_csv('data/full_cambodge_data.csv')
dfs = [df_guinee,df_cambodge]


# In[92]:


for HEIGHT_SIZE in HEIGHTS:
    composed = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((HEIGHT_SIZE,HEIGHT_SIZE)),
                               Pad(stride=8, pad_value=(0, 0, 0)),
                               normalization
                              ])
    for df in dfs:
        for img_path in df.img.values:
            img = io.imread(img_path)
            composed_ = composed(img).numpy().transpose(1,2,0)
            splitted = img_path.split('.')
            io.imsave(f'{splitted[0]}_{HEIGHT_SIZE}.{splitted[1]}',composed)

