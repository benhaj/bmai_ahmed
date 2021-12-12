import numpy as np
import torch
import os
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim
from torch import nn
import pandas as pd
import wandb

# start a new experiment


class BmaiTrainer:
    def __init__(self, model, dataset, seed = 0,img_size=256 ,AGE=False, SEXE=False, train_size=0.8, lr = 0.005, epochs = 20, batch_size=32, num_workers=10 ):
        """
        Initializes the class Kather19Trainer which inherits from the parent class Trainer. The class implements a
        convenient way to log training metrics and train over multiple sessions.
        :param model: currently trained model.
        :param loader: data loader for the training data.
        :param savepath: location where to save the different attributes of the class.
        :param optimizer: instantiated torch.optim optimizer used to train the model.
        :param scheduler: instantiated torch.optim.lr_scheduler used to reduce the learning rate.
        :param queue_length: number of batch of pre-computed embeddings (z) to store. Note that we store the
            embeddings (z), and not the similarities (C * z).
            If queue_length is None then no queue is used. In that case the batch size must be at least as large as
            the number of centroids/prototypes.
        :param loadpath: location where to load the class's attributes.
        """
        # Device
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        # Model
        self.img_size = img_size
        if model.name == 'mobilenet_v2':
            # Freeze parameters
            for param in model.parameters():
                param.requires_grad = False

            # Instantiate new fully connected layer
            if self.img_size==256:
                model.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=1280*8*8, out_features=256),
                    nn.ReLU(),
                    nn.Linear(in_features=256,out_features=128),
                    nn.ReLU(),
                    nn.Linear(128,32),
                    nn.ReLU()

                )
            else:
                model.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=1280*12*12, out_features=256),
                    nn.ReLU(),
                    nn.Linear(in_features=256,out_features=128),
                    nn.ReLU(),
                    nn.Linear(128,32),
                    nn.ReLU()
                )
            
            if AGE and SEXE:
                model.last = nn.Sequential(
                    #nn.Linear(in_features=256+2,out_features=128),
                    #nn.ReLU(),
                    #nn.Linear(128,32),
                    #nn.ReLU(),
                    nn.Linear(32+2,2)
                )
            elif AGE or SEXE:
                model.last = nn.Sequential(
                    #nn.Linear(in_features=256+1,out_features=128),
                    #nn.ReLU(),
                    #nn.Linear(128,32),
                    #nn.ReLU(),
                    nn.Linear(32+1,2)
                )
            else:
                model.last = nn.Sequential(
                    #nn.Linear(in_features=256+0,out_features=128),
                    #nn.ReLU(),
                    #nn.Linear(128,32),
                    #nn.ReLU(),
                    nn.Linear(32,2)
                )

        self.model = model.to(self.device)
        
        # Learning rate
        self.lr = lr

        self.epochs = epochs

        # Data generators
        self.dataset = dataset
        self.AGE = AGE
        self.SEXE = SEXE
        
        # Split data (train/test)
        self.seed = seed
        self.train_size = train_size
        
        num_train_entries = int(self.train_size * len(self.dataset))
        num_test_entries = len(self.dataset) - num_train_entries
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [num_train_entries, num_test_entries],generator=torch.Generator().manual_seed(self.seed))
        
        # Data loaders :
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)

        # Loss function and stored losses
        self.losses = []
        self.batch_losses = []

        

    def loss_fn(self,y_pred,y_true):
        diff = torch.abs(y_pred-y_true)
        return torch.where(diff < (0.05*y_true),torch.zeros(1, 2,dtype=float).to(self.device),diff)

    
    def train(self):
        """
        :param epochs: number of iterations over the training dataset to perform.
        :return: None.
        """
        AGE = self.AGE
        SEXE = self.SEXE

        print(AGE)
        print(SEXE)
        
        print(f'About to train for {self.epochs} epochs')
        model = self.model
        model.train()
        ## create optimizer
        optimizer = optim.Adam(model.parameters(),lr=self.lr)
        
        # capture a dictionary of hyperparameters with config
        wandb.config = {"learning_rate": self.lr, "epochs": self.epochs, "batch_size": self.batch_size}
        # optional: track gradients
        wandb.watch(model)


        results = pd.DataFrame(columns=['mean_height_rel_error','mean_weight_rel_error'])
        epoch_losses=[]
        for epoch in range(self.epochs):

            batch_losses = []
            for batch_idx, data in enumerate(self.train_dataloader):
                
                
                 ## data in form ['img',sexe','days','height','weight']
                
                imgs = data[0].to(self.device)
                target = data[1:][0]
                num_elems_in_batch = target.shape[0]
                ## Forward
                if AGE & SEXE:
                    sexe = target[:,0].reshape((num_elems_in_batch,1)).to(self.device)
                    age = target[:,1].reshape((num_elems_in_batch,1)).to(self.device)
                    target = target[:,2:].to(self.device)
                    if model.name == 'mobilenet_v2':
                      feat = model.classifier(model.features(imgs))
                      concat = torch.cat([feat,sexe,age],dim=1).float()
                      scores = model.last(concat)
                    else:
                      scores = model(imgs,sexe,age)
                    
                elif AGE:
                    age = target[:,0].reshape((num_elems_in_batch,1)).to(self.device)
                    target = target[:,2:].to(self.device)
                    if model.name == 'mobilenet_v2':
                      feat = model.classifier(model.features(imgs))
                      concat = torch.cat([feat,age],dim=1).float()
                      scores = model.last(concat)
                    else:  
                      scores = model(imgs,age)
                    
                elif SEXE:
                    sexe = target[:,1].reshape((num_elems_in_batch,1)).to(self.device)
                    target = target[:,2:].to(self.device)
                    if model.name == 'mobilenet_v2':
                      feat = model.classifier(model.features(imgs))
                      concat = torch.cat([feat,sexe],dim=1).float()
                      scores = model.last(concat)
                    else:
                      scores = model(imgs,sexe)
                    
                else:
                    target = target[:,2:].to(self.device)
                    feat = model.features(imgs)
                    classi = model.classifier(feat)
                    scores = model.last(classi)
                
                # loss
                loss = self.loss_fn(scores,target).sum()
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                
                # optimizer step
                optimizer.step()              
                
                batch_losses.append(loss.item())
                # Display status
                if batch_idx % 10 == 0:
                    message = f'epoch: {epoch}/{self.epochs-1}, batch {batch_idx}/{len(self.train_dataloader)-1}, loss: {loss.item()}' 
                    print(message)
            epoch_loss = np.mean(batch_losses)
            
            # log metrics inside your training loop to visualize model performance
            wandb.log({'epoch':epoch,"epoch_loss": epoch_loss})
            
            print(f'for epoch {epoch} , average loss is {epoch_loss}')
            epoch_losses.append(epoch_loss)
            
            mean_height_rel_error, mean_weight_rel_error, avg_test_loss = self.test(epoch)

            results.loc[epoch] = [mean_height_rel_error,mean_weight_rel_error]

        best_rel_err = (results.mean_height_rel_error.min(),results.mean_weight_rel_error.min())

        return best_rel_err, results, np.mean(epoch_losses)
    
    
    def test(self,epoch_num):
        """
        :param epochs: number of iterations over the training dataset to perform.
        :return: None.
        """
        AGE = self.AGE
        SEXE = self.SEXE
        print(f'About to test')
        model = self.model
        model.eval()
        batch_losses = []
        y_true = []
        predictions = []
        for batch_idx, data in enumerate(self.test_dataloader):
                

             ## data in form ['img',sexe','days','height','weight']

            imgs = data[0].to(self.device)
            target = data[1:][0]
            num_elems_in_batch = target.shape[0]                ## Forward
            if AGE & SEXE:
                sexe = target[:,0].reshape((num_elems_in_batch,1)).to(self.device)
                age = target[:,1].reshape((num_elems_in_batch,1)).to(self.device)
                target = target[:,2:].to(self.device)
                if model.name == 'mobilenet_v2':
                  feat = model.classifier(model.features(imgs))
                  concat = torch.cat([feat,sexe,age],dim=1).float()
                  scores = model.last(concat)
                else:
                    scores = model(imgs,sexe,age)

            elif AGE:
                age = target[:,0].reshape((num_elems_in_batch,1)).to(self.device)
                target = target[:,2:].to(self.device)
                if model.name == 'mobilenet_v2':
                    feat = model.classifier(model.features(imgs))
                    concat = torch.cat([feat,age],dim=1).float()
                    scores = model.last(concat)
                else:  
                    scores = model(imgs,age)

            elif SEXE:
                sexe = target[:,1].reshape((num_elems_in_batch,1)).to(self.device)
                target = target[:,2:].to(self.device)
                if model.name == 'mobilenet_v2':
                  feat = model.classifier(model.features(imgs))
                  concat = torch.cat([feat,sexe],dim=1).float()
                  scores = model.last(concat)
                else:
                    scores = model(imgs,sexe)

            else:
                target = target[:,2:].to(self.device)
                feat = model.features(imgs)
                classi = model.classifier(feat)
                scores = model.last(classi)

            y_true.append(target.detach().numpy() if self.device=='cpu' else target.cpu().detach().numpy())
            predictions.append(scores.detach().numpy() if self.device=='cpu' else scores.cpu().detach().numpy())

             # loss
            loss = self.loss_fn(scores,target).sum()

            batch_losses.append(loss.item() if self.device=='cpu' else loss.cpu().item())

                
        average_loss = np.mean(batch_losses)
        print(f'Average test loss is {average_loss}')
        
        y_true= np.vstack(y_true)
        predictions = np.vstack(predictions)
        
        mean_height_rel_error,mean_weight_rel_error = calculate_mean_absolute_error_results(y_true,predictions)
        print(f'mean_height_rel_error = {mean_height_rel_error}')
        print(f'mean_weight_rel_error = {mean_weight_rel_error}')
        
        wandb.log({'epoch':epoch_num,'epoch_test_loss':average_loss, 'mean_height_rel_error':mean_height_rel_error, 'mean_weight_rel_error':mean_weight_rel_error})

        return mean_height_rel_error,mean_weight_rel_error,average_loss




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
