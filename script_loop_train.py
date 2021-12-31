#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import pandas as pd

pd.DataFrame(columns=[
       'data',
       'img_size',
       'sexe',
       'age',
       'method_sex_age',
       'seed',
       'epochs',
       'lr',
       'height_rel_err',
       'weight_rel_err']).to_csv('results/Mobilenet_v2_full_results.csv',index=False)


data = ['guinee','cambodge','guinee_cambodge']
AGEs = ['False','True']
SEXEs =['False','True']

IMG_SIZEs = [256,384]

for data_name in data:
    for img_size in IMG_SIZEs:
        for AGE in AGEs:
            for SEXE in SEXEs:
                for SEED in [0,1,2,3]:
                    os.system(f'python demo_bmai.py --model_name mobilenet --data_name {data_name} --SEED {SEED} --img_size {img_size} --SEXE {SEXE} --AGE {AGE} --method_sex_age 0 --epochs 30 --lr 0.005 --batch_size 128 --num_workers 16')
