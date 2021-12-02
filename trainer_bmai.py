import numpy as np
import torch
import os
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim


class BmaiTrainer:
    def __init__(self, model, dataset, AGE=False, SEXE=False, train_size=0.8, lr = 0.005, batch_size=32, num_workers=0 ):
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

        # Model
        self.model = model.to(self.device)
        
        # Learning rate
        self.lr = lr 

        # Data generators
        self.dataset = dataset
        self.AGE = AGE
        self.SEXE = SEXE
        
        # Split data (train/test)
        self.train_size = train_size
        
        num_train_entries = int(self.train_size * len(self.dataset))
        num_test_entries = len(self.dataset) - num_train_entries
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [num_train_entries, num_test_entries])
        
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

    
    def train(self, epochs):
        """
        :param epochs: number of iterations over the training dataset to perform.
        :return: None.
        """
        AGE = self.AGE
        SEXE = self.SEXE
        print(f'About to train for {epochs} epochs')
        model = self.model
        ## create optimizer
        optimizer = optim.Adam(model.parameters(),lr=self.lr)
        
        epoch_losses=[]
        for epoch in range(epochs):
            #self.model.train()
            batch_losses = []
            for batch_idx, data in enumerate(self.train_dataloader):
                
                
                 ## data in form ['img',sexe','days','height','weight']
                
                imgs = data[0].to(self.device)
                target = data[1:][0]
                ## Forward
                if AGE & SEXE:
                    sexe = target[:,0].to(self.device)
                    age = target[:,1].to(self.device)
                    target = target[:,2:].to(self.device)
                    
                    scores = model(imgs,sexe,age)
                    
                elif AGE:
                    age = target[:,0].to(self.device)
                    target = target[:,2:].to(self.device)
                    
                    scores = model(imgs,age)
                    
                elif SEXE:
                    sexe = target[:,1].to(self.device)
                    target = target[:,2:].to(self.device)
                    
                    scores = model(imgs,sexe)
                    
                else:
                    target = target[:,2:].to(self.device)
                    scores = model(imgs)
                
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
                    message = f'epoch: {epoch}/{epochs-1}, batch {batch_idx}/{len(self.train_dataloader)-1}, loss: {loss.item()}' 
                    print(message)
            epoch_loss = np.mean(batch_losses)
            print(f'for epoch {epoch} , average loss is {epoch_loss}')
            epoch_losses.append(epoch_loss)
        return epoch_losses
    
    
    def test(self):
        """
        :param epochs: number of iterations over the training dataset to perform.
        :return: None.
        """
        AGE = self.AGE
        SEXE = self.SEXE
        print(f'About to test')
        model = self.model
        
        batch_losses = []
        y_true = []
        predictions = []
        for batch_idx, data in enumerate(self.test_dataloader):


             ## data in form ['img',sexe','days','height','weight']

            imgs = data[0].to(self.device)
            target = data[1:][0]
            
            ## Forward
            if AGE & SEXE:
                sexe = target[:,0].to(self.device)
                age = target[:,1].to(self.device)
                target = target[:,2:].to(self.device)

                scores = model(imgs,sexe,age)

            elif AGE:
                age = target[:,1].to(self.device)
                target = target[:,2:].to(self.device)

                scores = model(imgs,age)

            elif SEXE:
                sexe = target[:,0].to(self.device)
                target = target[:,2:].to(self.device)

                scores = model(imgs,sexe)

            else:

                target = target[:,2:].to(self.device)
                scores = model(imgs)
            
            y_true.append(target.detach().numpy())
            predictions.append(scores.detach().numpy())
            
            # loss
            loss = self.loss_fn(scores,target).sum()
            
            batch_losses.append(loss.item())
            
                
        average_loss = np.mean(batch_losses)
        print(f'Average test loss is {average_loss}')
        
        y_true= np.vstack(y_true)
        predictions = np.vstack(predictions)
        
        
        return y_true,predictions,average_loss
