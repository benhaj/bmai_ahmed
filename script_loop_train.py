#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import pandas as pd

# pd.DataFrame(columns=[
#        'data',
#        'img_size',
#        'sexe',
#        'age',
#        'method_sex_age',
#        'seed',
#        'epochs',
#        'lr',
#        'height_rel_err',
#        'weight_rel_err']).to_csv('results/mobilenet_v2_results_with_age_sexe_branch.csv',index=False)


data = ['guinee','cambodge','guinee_cambodge']
AGEs = ['False','True']
SEXEs =['False','True']

IMG_SIZEs = [256,384]

for data_name in data[:2]:
    for img_size in IMG_SIZEs[1:]:
        for AGE in AGEs[1:]:
            for SEXE in SEXEs[1:]:
                for SEED in [0,1,2,3]:
                    if data_name=='guinee':
                        epochs=30
                        os.system(f'python demo_bmai.py --model_name mobilenet --data_name {data_name} --SEED {SEED} --img_size {img_size} --SEXE {SEXE} --AGE {AGE} --method_sex_age 4 --epochs {epochs} --lr 0.005 --batch_size 128 --num_workers 16')
                    else:
                        epochs=30
                        os.system(f'python demo_bmai.py --model_name mobilenet --data_name {data_name} --SEED {SEED} --img_size {img_size} --SEXE {SEXE} --AGE {AGE} --method_sex_age 4 --epochs {epochs} --lr 0.005 --batch_size 128 --num_workers 16')
