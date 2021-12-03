#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os

data = ['cambodge']
AGEs = [False,True]
SEXEs =[False,True]

IMG_SIZEs = [256]

for data in data:
    for AGE in AGEs:
        for SEXE in SEXEs:
            for size in IMG_SIZEs:
                os.system(f'python demo_bmai.py --model_name mobilenet --data_name {data} --img_size {size} --SEXE {SEXE} --AGE {AGE} --epochs 20 --lr 0.005 --batch_size 128 --num_workers 16')

