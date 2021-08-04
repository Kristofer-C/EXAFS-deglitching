# -*- coding: utf-8 -*-
"""
Creates a LSTM to take in chunk of values from a sequence and predict the next
value in the sequence.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import time


def chunk_sequence(sequence, chunk_size=10, n_leave=0):
    """
    Forms a tensor of chunks of consecutive values from the sequence.
    """
    
    chunks=[]
    for i in range(0,len(sequence)-chunk_size+1-n_leave):
        chunks.append(sequence[i:i+chunk_size])
        
    return torch.stack(chunks)


# Turns out that LSTM training is batchable
# So here's a standard dataset class
# The transform is not optional because it is necessary
# for the signals to be in a different form for the LSTM
class CleanData(Dataset):
    
    def __init__(self,
                 signals_dir,
                 #signals_file,
                 chunk_size,
                 transform,
                 device):
        
        self.signals_dir=signals_dir
        self.transform=transform
        self.chunk_size=chunk_size
        self.chilist=os.listdir(signals_dir)
        self.device=device
        
        #self.dat=np.genfromtxt(signals_dir)
        
    def __len__(self):
        
        return len(self.chilist)
        #return len(self.dat)
    
    
    def __getitem__(self, idx):
        
        signal=torch.from_numpy(np.genfromtxt(os.path.join(self.signals_dir, 
                                            self.chilist[idx])))[50:]
        #signal=torch.from_numpy(self.dat[idx])
        chunks, labels=self.transform(signal)
            
        return chunks, labels
    
    
class NoiseAndChunkTransform:
    """
    Normalizes the spectrum, forms the seuqnce into chunks of
    consecutive values, and returns the tensor of chunks and the tensor
    of labels (values to be predicted by the model).
    """
    
    def __init__(self, chunk_size, noise_std, device):
        self.chunk_size=chunk_size
        self.noise_std=noise_std
        self.device=device
        
    def __call__(self, y):
        
        # Enforce the the tensor has values with a range of 1
        #y/=(max(y)-min(y))
        
        # Add a small amount of noise
        #y=y[50:]
        #x=np.arange(len(y))
        #a, b = np.polyfit(x, y, 1)
        #y-=a*x+b
        #y/=max(y)-min(y)
        noisy=y+(torch.randn(len(y))*self.noise_std)#.to(self.device)
        
        # Form the chunks, leaving one value at the end for the last prediction
        chunks=chunk_sequence(noisy, self.chunk_size, 1)
        
        # The labels are just the value that immediately follows the chunk
        labels=y[chunk_size:]

        return chunks, labels



class LSTMModel(nn.Module):
    def __init__(self, 
                 in_size, 
                 hidden_size, 
                 out_size,
                 batch_size,
                 num_layers, 
                 drop_prob, 
                 bidirectional,
                 device,
                 batch_first=True):
        super().__init__()
        
        self.device=device
        
        self.hidden_size = hidden_size
        self.out_size=out_size
        self.batch_size=batch_size # Necessary for initiating the hidden state
        self.num_layers=num_layers
        self.bidirectional=int(bidirectional)
        
        self.lstm = nn.LSTM(in_size, 
                            hidden_size, 
                            num_layers=num_layers, 
                            dropout=drop_prob,
                            bidirectional=bidirectional,
                            batch_first=batch_first)
        
        self.conv=nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(in_size//2)
            )
        
        self.linear = nn.Sequential(
                nn.Dropout(p=drop_prob),
                nn.Linear((1+self.bidirectional)*self.hidden_size, 16),
                nn.Tanh(),
                #nn.ReLU(),
                nn.Linear(16, self.out_size))
        
        self.init_hidden(batch_size)
        
    def forward(self, x):
        
        #out = self.conv(x.view(len(x), len(x[0]), 1, -1)).squeeze()
        
        out, self.hidden_state = self.lstm(
                                    x, self.hidden_state)
        
        self.hidden_state = tuple(
            [h.detach() for h in self.hidden_state])
        
        out = self.linear(out)
        return out.squeeze()+x[:,:,-2:].mean(dim=-1)
    
    # self.hidden_state contains both the hidden state and the cell state
    # of the LSTM cell
    def init_hidden(self, batch_size):
        
        self.hidden_state = (
            torch.zeros(((1+self.bidirectional)*self.num_layers,
                         batch_size,
                         self.hidden_size)).to(self.device),
            torch.zeros(((1+self.bidirectional)*self.num_layers, 
                         batch_size, 
                         self.hidden_size)).to(self.device))
        
        
    




def train_loop(dataloader, model, loss_func, optimizer, device):
    """
    Trains the model with one iteration through the entire dataset in batches.
    Returns the average of the losses from each batch.
    """
    
    size=len(dataloader.dataset)
    losses=[]
    
    for batch, (X, lb) in enumerate(dataloader):
        
        model.init_hidden(len(X))
        X, lb = X.to(device), lb.to(device)
        prediction=model(X.float())
        loss=loss_func(prediction.squeeze(), lb.float().squeeze())
        
        # Take a step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss=loss.item()#/len(X)
        
        # Every 25 batches, print the status
        if (batch+1)%max(1, len(dataloader)//10)==0:
            current = batch*len(X)
            print("Loss: %.4f, %d/%d"%(loss, current, size))
            
        losses.append(loss)
        
    return np.array(losses).mean()
            
            
def test_loop(dataloader, model, loss_func, device):
    size=len(dataloader.dataset)
    test_loss=0
    
    """
    Tests the model with one iteration through the entire testing set in batches.
    Returns the average testing loss.
    """
    
    with torch.no_grad():
        count=0
        for X, lb in dataloader:
            model.init_hidden(len(X))
            #X=X.float()
            X, lb = X.to(device), lb.to(device)
            prediction=model(X.float())               
            test_loss += loss_func(prediction.squeeze(), lb.float().squeeze()).item()
            count+=1
    test_loss/=count
    
    print("Current testing loss: %.4f"%test_loss)
    return test_loss

        
if __name__=="__main__":
    
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device='cpu'
    print("Device: %s"%device)
    
    #train_file = "chi_train.txt"
    #valid_file = "chi_valid.txt"
    
    train_dir = "mu_train"
    valid_dir = "mu_validation"
    
    # The number of points used to make a prediction on the next point/
    # update the cell state. I'm hesitant about increasing this because it
    # means that the model won't be able to start making predictions until
    # even later into the signal
    chunk_size=16
    # The size of the noise added to the training data
    noise_std=0.01
    
    train_dat=CleanData(train_dir, 
                        chunk_size, 
                        NoiseAndChunkTransform(chunk_size, noise_std, device),
                        device)
    valid_dat=CleanData(valid_dir, 
                       chunk_size, 
                       NoiseAndChunkTransform(chunk_size, noise_std, device),
                       device)
    
    batch_size=64
    train_dl = DataLoader(train_dat, 
                          batch_size=batch_size, 
                          shuffle=True)
    
    valid_dl = DataLoader(valid_dat, 
                         batch_size=batch_size, 
                         shuffle=True)
    

    # The length of the cell state and hidden state vectors in the LSTM cell
    hidden_size=32
    # The number of next values in the sequence to predict
    out_size=1

    bidirectional=False
    num_layers=4
    drop_prob=0.5
    model=LSTMModel(chunk_size, 
                    hidden_size, 
                    out_size, 
                    batch_size,
                    num_layers, 
                    drop_prob,
                    bidirectional,
                    device).to(device)
    
    


    loss_func=nn.MSELoss(reduction="mean")
    
    learning_rate=1e-2
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.7)
    lambda1 = lambda epoch: np.exp(-epoch/10)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
    #                                                max_lr=1e-2,
    #                                                steps_per_epoch=10,
    #                                                epochs=epochs)
    
    epochs=10
    
    train_losses=[]
    test_losses=[]
    
    t0=time.time()
    t=t0
    
    for epoch in range(epochs):
        print("===================================================")
        print("Epoch: %d"%(epoch+1))
        
        train_loss=train_loop(train_dl, model, loss_func, optimizer, device)
        test_loss=test_loop(valid_dl, model, loss_func, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        scheduler.step()
        
        ttemp=time.time()
        print("Time for epoch: %.3f seconds"%(ttemp-t))
        t=ttemp
        
        print("===================================================")
    
        
    print("Done!")
    print("Total time: %.3f seconds"%(t-t0))
        
    torch.save(model.state_dict(), "lstm_mu2.pth")

    epo=list(range(1,epochs+1))
    plt.plot(epo, train_losses, epo, test_losses)


