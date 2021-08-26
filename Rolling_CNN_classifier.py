"""
Trains a sliding CNN anomaly detector.
"""
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class ChunksData(Dataset):
    
    def __init__(self,
                 fname,
                 transform=None):
        
        self.fname=fname
        self.transform=transform
        self.data=torch.Tensor(np.genfromtxt(self.fname, dtype="float")[::31])
        
        
    def __len__(self):
        
        return len(self.data)
    
    
    def __getitem__(self, idx):
        
        row=self.data[idx]
        
        chunk, label = row[1:], (row[0]).long()
        #label=torch.eye(3)[label]     
        
        if self.transform:
            chunk=self.transform(chunk)
            
        return chunk, label
    
    
class NoiseAndNormTransform:
    """
    Normalizes the chunks and adds noise, forms the seuqnce into chunks of
    consecutive values, and returns the tensor of chunks and the tensor
    of labels (values to be predicted by the model).
    """
    
    def __init__(self, noise_std):
        self.noise_std=noise_std

        
    def __call__(self, chunk):
        
        
        # Add a small amount of noise
        chunkT=chunk+torch.randn(len(chunk))*self.noise_std#*torch.rand(1)
        # Normalize the chunk
        chunkT=(chunkT-min(chunkT))/(max(chunkT)-min(chunkT))
        
        return chunkT
    
    
class RollingCNN(nn.Module):
    
    def __init__(self, out_size):
        super(RollingCNN, self).__init__()
        
        # End up with 256 activation maps chunk_size points long
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU())#,
            #nn.Conv1d(256, 512, kernel_size=3, padding=1),
            #nn.ReLU())

        
        # Take the highest value from all these maps and flatten it to a 256x1
        # vector
        # Take the vector and map it to 3 values to predict the type of glitch
        # (if any) in the chunk
        self.lin_stack = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.Tanh(),
            #nn.ReLU(),
            nn.Linear(32, out_size))#,
            #nn.ReLU()) 
        
    def forward(self, x):
        out=self.conv_stack(x.view(len(x), 1, len(x[0])))
        out=self.lin_stack(out)
        return out.squeeze() # Return the batch predictions in a form compatible
                            # with the loss function
        
        
def train_loop(dataloader, model, loss_func, optimizer, device):
    """
    Trains the model with one iteration through the entire dataset in batches.
    Returns a list of losses from each batch.
    """
    
    size=len(dataloader.dataset)
    losses=[]
    
    for batch, (X, lb) in enumerate(dataloader):
        X, lb = X.to(device), lb.to(device)
        prediction=model(X)
        loss=loss_func(prediction, lb)
        
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
    Returns the average testing loss and prints the confusion matrix. The confusion
    matrix is organized so that each row corresponds to a unique label, and the 
    columns are the distribution of predictions. 
    Class 0: No glitch
    Class 1: Monochromator glitch
    Class 2: Step glitch.
    """
    
    
    # The identity matrix maps the classification number (0, 1, or 2) to a vector
    # with a 1 in that index.
    # The one-hot vectors are used to construct the confusion matrix.
    binmap=torch.eye(3)
    confusion=torch.zeros((3,3))
    
    with torch.no_grad():
        count=0
        for X, lb in dataloader:
            #X=X.float()
            X, lb = X.to(device), lb.to(device)
            prediction=model(X)               
            test_loss += loss_func(prediction, lb).item()
            
            pbin=binmap[torch.argmax(prediction, 1)]
            lbin=binmap[lb]
            confusion+=sum(torch.bmm(lbin.unsqueeze(2), pbin.unsqueeze(1)))            
            
            count+=1
            
    
    #print("Recent predictions: ")        
    #print(prediction[0:5])
    test_loss/=count
    print(confusion.int())
    return test_loss


if __name__=="__main__":
    
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device='cpu'
    print("Device: %s"%device)
    train_file="glitchy_chunks_mu_train.txt"
    valid_file="glitchy_chunks_mu_validation.txt"
    out_size=3
    batch_size=256
    noise_std=0.005
    
    train_dat=ChunksData(train_file, NoiseAndNormTransform(noise_std))
    valid_dat=ChunksData(valid_file, NoiseAndNormTransform(noise_std))
    
    
    train_dl = DataLoader(train_dat, 
                          batch_size=batch_size, 
                          shuffle=True)
    
    valid_dl = DataLoader(valid_dat, 
                         batch_size=batch_size, 
                         shuffle=True)
    
    #loss_func=nn.BCELoss()
    loss_func=nn.CrossEntropyLoss() # For multiclass classification
    
    model=RollingCNN(out_size).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
                                 weight_decay=1e-4)
    
    lambda1 = lambda epoch: np.exp(-epoch/5)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    
    
    epochs=10
    print("Training set: %s\nTesting set: %s"%(train_file, valid_file))
    
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
        print("Current testing loss: %.4f"%test_loss)
        
        scheduler.step()
        
        ttemp=time.time()
        print("Time for epoch: %.3f seconds"%(ttemp-t))
        t=ttemp
        
        print("===================================================")
    
        
    print("Done!")
    print("Total time: %.3f seconds"%(t-t0))
        
    #torch.save(model.state_dict(), "Rolling_glitch_classifier1.pth")

    epo=list(range(1,epochs+1))
    plt.plot(epo, train_losses, epo, test_losses)