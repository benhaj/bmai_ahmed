#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import argparse
from prepare_dataset import *
from trainer_bmai_2 import *
import torchvision
import torch


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='''Bmai human height and width estimation using mobilenet''', add_help=False)
    parser.add_argument('--model_name', default='mobilenet', type=str,
                        help='model to use')
    parser.add_argument('--data_name', type=str, default='guinee', help='dataset to use (either "guinee" or "cambodge")')
    parser.add_argument('--img_size',type=int,default=256,help='size of images (either 256 or 386)')
    parser.add_argument('--SEXE', type=bool, default=False, help='use sexe attribute ?')
    parser.add_argument('--AGE', type=bool, default=False, help='use age attribute ?')
    parser.add_argument('--epochs', type=int, default=15, help='how many training epochs')
    parser.add_argument('--lr',type=float,default=0.005,help='learning rate')
    parser.add_argument('--batch_size',type=int,default=32,help='batch size')
    parser.add_argument('--num_workers',type=int,default=8,help='number of workers')
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
        dataset = bmaiDataset(csv_file='/hdd/data/bmai_clean/full_guinee_data.csv',img_size=args.img_size,transform=transforms)
    else:
        dataset = bmaiDataset(csv_file='/hdd/data/bmai_clean/full_cambodge_data.csv',img_size=args.img_size,transform=transforms)
        
    ## Trainer:
    trainer = BmaiTrainer(model, dataset, img_size=args.img_size, AGE=args.AGE, SEXE=args.SEXE, batch_size=args.batch_size, lr = args.lr, num_workers=args.num_workers)
    results,mean_training_loss = trainer.train(args.epochs)
    torch.save(model,f'{args.model_name}_{args.data_name}_{args.img_size}_SEXE_{args.SEXE}_AGE_{args.AGE}_{args.epochs}_epochs.pt')
    results.to_csv(f'{args.model_name}_{args.data_name}_{args.img_size}_SEXE_{args.SEXE}_AGE_{args.AGE}_{args.epochs}_epochs.csv')
    #y_true,predictions,average_loss = trainer.test()

if __name__ == '__main__':
    ## parser = argparse.ArgumentParser('Demo', parents=[get_args_parser()])
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

