#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os

data = ['guinee','cambodge','guinee_cambodge']
AGEs = ['False','True']
SEXEs =['False','True']

IMG_SIZEs = [256,384]

for data_name in data[1:2]:
    for img_size in IMG_SIZEs[:1]:
        for AGE in AGEs[:1]:
            for SEXE in SEXEs[:1]:
                for SEED in [0,1,2,3][:1]:
                    os.system(f'python demo_bmai.py --model_name mobilenet --data_name {data_name} --SEED {SEED} --img_size {img_size} --SEXE {SEXE} --AGE {AGE} --epochs 30 --lr 0.005 --batch_size 128 --num_workers 16')
