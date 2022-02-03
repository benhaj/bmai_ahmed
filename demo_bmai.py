#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import argparse
from prepare_dataset import *
from source.models.mobilenet_bmai import *
from trainer_bmai_2 import *
import torchvision
import torch
from source.models.OpenPose_bmai import *


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='''Bmai human height and width estimation using mobilenet''', add_help=False)
    parser.add_argument('--model_name', default='mobilenet', type=str,
                        help='model to use')
    parser.add_argument('--data_name', type=str, default='guinee', help='dataset to use (either "guinee" or "cambodge")')
    parser.add_argument('--SEED',type=int,default=0,help='SEED to perform validation')
    parser.add_argument('--img_size',type=int,default=256,help='size of images (either 256 or 386)')
    parser.add_argument('--SEXE', type=str, default='False', help='use sexe attribute ?')
    parser.add_argument('--AGE', type=str, default='False', help='use age attribute ?')
    parser.add_argument('--method_sex_age', type=int, default=None, help='How to use age and sex attributes ? (see documentation of Mobilenet_bmai class')
    parser.add_argument('--epochs', type=int, default=15, help='how many training epochs')
    parser.add_argument('--lr',type=float,default=0.005,help='learning rate')
    parser.add_argument('--batch_size',type=int,default=32,help='batch size')
    parser.add_argument('--num_workers',type=int,default=8,help='number of workers')
    return parser


def main(args):

    if (args.SEXE == 'False'):
        SEXE = False
    else:
        SEXE = True

    if (args.AGE == 'False'):
        AGE = False
    else:
        AGE = True


    ## MODEL
    if args.model_name == 'mobilenet':
        model = Mobilenet_bmai(args.img_size,SEXE=SEXE, AGE=AGE, method_sex_age = args.method_sex_age)
    else:
        model = prepare_OpenPose_model(freeze=True,method_age_sex=args.method_sex_age)
    
    ## DATA
    dataset = prepare_dataset(args.data_name,args.img_size)
        
    ## TRAINER:
    
    #wandb initialization
    run_name =f'{args.model_name}_Full_{args.data_name}_SEED{args.SEED}_{args.img_size}_SEXE_{SEXE}_AGE_{AGE}_method_{args.method_sex_age}_{args.epochs}_epochs_lr_{args.lr}'
    wandb.init(project="new-sota-model",name=run_name)

    trainer = BmaiTrainer(model, dataset, seed=args.SEED, img_size=args.img_size, AGE=AGE, SEXE=SEXE, method_sex_age=args.method_sex_age , batch_size=args.batch_size, lr = args.lr, epochs=args.epochs, num_workers=args.num_workers)
    best_ , results,mean_training_loss = trainer.train()
#     torch.save(trainer.model,f'results/best_model_mobilenet_v2_cambodge.pt')
#     results.to_csv(f'results/{run_name}.csv',index=False)

#     summary_df = pd.json_normalize(create_df_entry(args,best_))
#     summary_df.to_csv(f'results/mobilenet_v2_results_with_age_sexe_in_regression.csv', mode='a', header=False,index=False)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

