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


# In[38]:


df_guinee = pd.read_csv('/hdd/data/bmai_clean/full_guinee_data.csv')
df_cambodge = pd.read_csv('/hdd/data/bmai_clean/full_cambodge_data.csv')
dfs = [df_guinee,df_cambodge]

total_length = len(df_guinee)+len(df_cambodge)

# In[92]:


for HEIGHT_SIZE in HEIGHTS[1:]:
    composed = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((HEIGHT_SIZE,HEIGHT_SIZE)),
                               Pad(stride=8, pad_value=(0, 0, 0))
                              ])

    print(f'Processing for size {HEIGHT_SIZE} ..')
    i = 0
    for df in dfs[:1]:
        for img_path_ in df.img.values:
            i=i+1

            img_path = "/hdd/data/" + img_path_.replace('data','bmai_clean',1)
            img = io.imread(img_path)
            composed_ = composed(img.copy()).numpy().transpose(1,2,0)
            splitted = img_path.split('.')
            if len(splitted)!=2:
                new_path = f'{splitted[0]}.{splitted[1]}_{HEIGHT_SIZE}.{splitted[2]}'
            else:    
                new_path = f'{splitted[0]}_{HEIGHT_SIZE}.{splitted[1]}'
            io.imsave(new_path,(composed_*255).astype(np.uint8))

            if i%(int(total_length*0.05)) == 0:
                print(f'{int(100*i/total_length)}% Done')

