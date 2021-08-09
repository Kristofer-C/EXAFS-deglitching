# -*- coding: utf-8 -*-
"""
Creates and trains a LSTM to take in chunks of values from a sequence and predict 
the next value in the sequence. It is currently set up to take in chi(k) EXAFS
data to be able to make next-point predictions on chi. This is intended to be 
used as an automatic glitch detection and correction algorithm.
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
    
    Parameters:
    sequence (float, pytorch tensor): The sequence of values to be chunked
    chunk_size (int): The number of points in each chunk.
    n_leave (int, default=0): The number of points in the sequence at the end to 
    leave out of the list of chunks.
    
    Returns:
    chunks (float, pytorch tensor): The list of chunks.
    """
    
    chunks=[]
    for i in range(0,len(sequence)-chunk_size+1-n_leave):
        chunks.append(sequence[i:i+chunk_size])
        
    return torch.stack(chunks)



class CleanData(Dataset):
    """
    A typical dataset class for training an algorithm. 
    
    __init__ initializes variables used in other functions of the class.
    
    __len__ returns the number of signals in the given directory
    
    __getitem__ returns the chunked signal and corresponding labels
    of the signal corresponding to index idx.
    """
    
    def __init__(self,
                 signals_dir,
                 chunk_size,
                 transform):
        """
        Initialize variables for the class. 
        
        signals_dir (str): The path to the folder containing each signal
        in a .txt file.
        transform (python class): The class used to transform the signals
        into chunks
        chunk_size (int): The number of points use to form each chunk.
        chilist (list, str): The full list of .txt file names containing the signals.
        """
        
        self.signals_dir=signals_dir
        self.transform=transform
        self.chunk_size=chunk_size
        self.chilist=os.listdir(signals_dir)
        
        
    def __len__(self):
        """Returns the number of available signals for training or testing
        in the given directory."""
        
        return len(self.chilist)

    
    def __getitem__(self, idx):
        """
        Loads the signal corresponding to the index idx in 
        the list of signals chilist, and forms it into chunks
        of size chunk_size. 
        
        Returns the list of chunks for that signal and the list
        of labels (of the same length as chunks) which are just
        the points in the signals immediately following the end
        of the chunks."""
        
        signal=torch.from_numpy(np.genfromtxt(os.path.join(self.signals_dir, 
                                            self.chilist[idx])))

        chunks, labels=self.transform(signal)
            
        return chunks, labels
    
    
class NoiseAndChunkTransform:
    """
    Normalizes the spectrum, adds an amount of gaussian noise,
    forms the seuqnce into chunks of consecutive values, 
    and returns the tensor of chunks and the tensor
    of labels (values to be predicted by the model).
    """
    
    def __init__(self, chunk_size, noise_std):
        self.chunk_size=chunk_size
        self.noise_std=noise_std
        
    def __call__(self, y):
        
        # Enforce the the tensor has values with a range of 1
        y/=(max(y)-min(y))
        
        # Add a small amount of noise
        noisy=y+(torch.randn(len(y))*self.noise_std)
        
        # Form the chunks, leaving one value at the end for the last prediction
        chunks=chunk_sequence(noisy, self.chunk_size, 1)
        
        # The labels are just the value that immediately follows the chunk
        labels=y[chunk_size:]
        
        return chunks, labels



class LSTMModel(nn.Module):
    """
    Defines the machine learning model that uses one or more LSTM cells.
    
    __init__ initializes values and the model structure.
    
    forward calls the model on a tensor.
    
    init_hidden initializes the hidden state of the LSTM to zeros.
    """
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
        """Initializes the model structure, and the values and parameters used to
        call the model.
        
        in_size (int): The number of points in a sequence used to generate an output 
        from the LSTM cell. In practice, this is equal to chunk_size.
        
        hidden_size (int): The number of points used in the hidden state of the LSTM cell
        
        out_size (int): The number of values to be output by the model in a single call.
        
        batch_size (int): The size of the batches that are to be used for training 
        and testing
        
        num_layers (int): The number of LSTM layers used in the LSTM cell.
        
        drop_prob (float in [0,1]): The fraction of of values in the hidden states
        that get randomly set to zero during a forward call.
        
        bidirectional (bool): If true, the LSTM cell becomes bidirectional
        
        device (str): The device on which the computations are being performed.
        
        batch_first (bool, default=True): Specifies whether the batch size is the 
        first dimension of the input tensor.
        
        lstm (nn module): The LSTM cell defined with the paramters above.
        
        linear (nn module): The fully connected linear layer that takes
        the hidden state in the LSTM and outputs out_size number of values.        
        """
        
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

        
        self.linear = nn.Sequential(
                # Option to include a dropout rate between the LSTM and the linear layers.
                #nn.Dropout(p=drop_prob),
                nn.Linear((1+self.bidirectional)*self.hidden_size, 16),
                nn.Tanh(),
                nn.Linear(16, self.out_size))
        
        self.init_hidden(batch_size)
        
        
    def forward(self, x):
        """Calls the model and returns the ouptut values."""
        
        out, self.hidden_state = self.lstm(
                                    x, self.hidden_state)
        
        self.hidden_state = tuple(
            [h.detach() for h in self.hidden_state])
        
        out = self.linear(out)
        
        # Add the average of the last two points in the chunk to the
        # output value so that the model doesn't have to predict
        # the actual next value in the sequence, but just the change
        # from the last two points. This makes the predictions more 
        # likely to be close to the previous value, which is expected
        # for smoothly changing data like EXAFS.
        return out.squeeze()+x[:,:,-2:].mean(dim=-1)
    
    # self.hidden_state contains both the hidden state and the cell state
    # of the LSTM cell
    def init_hidden(self, batch_size):
        """Sets the hidden and cell states of the LSTM cell to zero."""
        
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
    # Keep track of the loss from each batch.
    losses=[]
    
    # Iterate through the batches
    # X is the tensor of chunked signals,
    # lb is the tensor with the corresponding labels
    for batch, (X, lb) in enumerate(dataloader):
        
        # Initialize the state of the LSTM to zero with the 
        # appropriate batch_size
        model.init_hidden(len(X))
        # Cast the chunks and labels to the device
        X, lb = X.to(device), lb.to(device)
        # Run the model, making predictions for the points following the chunks
        prediction=model(X.float())
        # Compute the loss between the predictions and the labels.
        loss=loss_func(prediction.squeeze(), lb.float().squeeze())
        
        # Take a step in the training.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss=loss.item()
        
        # After every tenth of the training set, print the status
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

    train_dir = "chi_train"
    valid_dir = "chi_validation"
    
    # The number of points used to make a prediction on the next point/
    # update the cell state. I'm hesitant about increasing this because it
    # means that the model won't be able to start making predictions until
    # even later into the signal
    chunk_size=16
    # The size of the noise added to the training data
    noise_std=0.01
    
    # Define the datasets for training and cross validation
    train_dat=CleanData(train_dir, 
                        chunk_size, 
                        NoiseAndChunkTransform(chunk_size, noise_std, device),
                        device)
    valid_dat=CleanData(valid_dir, 
                       chunk_size, 
                       NoiseAndChunkTransform(chunk_size, noise_std, device),
                       device)
    
    # Create the pytorch dataloaders
    batch_size=64
    train_dl = DataLoader(train_dat, 
                          batch_size=batch_size, 
                          shuffle=True)
    
    valid_dl = DataLoader(valid_dat, 
                         batch_size=batch_size, 
                         shuffle=True)
    
    # Define the model
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
    
    
    # Use the mean squared error as a loss function
    loss_func=nn.MSELoss(reduction="mean")
    
    # Define the optimizer and scheduler
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
    
    # Keep track of the training and testing losses
    train_losses=[]
    test_losses=[]
    
    # Keep track of how long each epoch and the whole training process takes.
    t0=time.time()
    t=t0
    
    # Run for epoch number of times through the training and testing sets.
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
       
    # Save the newly trained model
    torch.save(model.state_dict(), "lstm_chi1.pth")

    # Plot the test and training losses as a function of the epochs.
    epo=list(range(1,epochs+1))
    plt.plot(epo, train_losses, epo, test_losses)


