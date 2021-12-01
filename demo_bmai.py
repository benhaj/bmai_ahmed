#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import argparse
from prepare_dataset import *
from trainer_bmai_2 import *
import torchvision


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='''Bmai human height and width estimation using mobilenet''', add_help=False)
    parser.add_argument('--model_name', default='mobilenet', type=str,
                        help='model to use')
    parser.add_argument('--data_name', type=str, default='guinee', help='dataset to use (either "guinee" or "cambodge")')
    parser.add_argument('--SEXE', type=bool, default=False, help='use sexe attribute ?')
    parser.add_argument('--AGE', type=bool, default=False, help='use age attribute ?')
    parser.add_argument('--epochs', type=int, default=15, help='how many training epochs')
    return parser


def main(args):
    
    ## Model
    if args.model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.name = 'mobilenet_v2'
    
    ## Data
    # TRANSFORMS :
    transforms = prepare_transforms()
    if args.data_name == 'guinee':
        dataset = bmaiDataset(csv_file='data/full_guinee_data.csv',transform=transforms,use_csv=True)
    else:
        dataset = bmaiDataset(csv_file='data/full_cambodge_data.csv',transform=transforms,use_csv=True)
        
    ## Trainer:
    trainer = BmaiTrainer(model, dataset, AGE=args.AGE, SEXE=args.SEXE, batch_size=32, num_workers=0)
    mean_training_loss = trainer.train(args.epochs)
    y_true,predictions,average_loss = trainer.test()

if __name__ == '__main__':
    ## parser = argparse.ArgumentParser('Demo', parents=[get_args_parser()])
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

