# Makes a glitchy signal and deglitches it using the LSTM model
# Some code may be added for deglitching with other methods or models

import numpy as np
import matplotlib.pyplot as plt
import torch
from LSTM_on_chi import *
from scipy.ndimage import gaussian_filter
import scipy.stats as stats


def add_point_glitch_clean_start(y, 
                                 num_glitches_range=2,
                                 min_size=0.2,
                                 max_size=0.5,
                                 skip_num_points=10,
                                 return_inds=False):
    """
    Adds a random number of jump glitches to a signal. Glitch locations are
    randomly chosen to be anywhere except at the beginning for a certain 
    number of points.
    
    PARAMETERS:
        y (array, float): The input signal to add glitches to.
        num_glitches_range (int): The maximum number of glitches to add
        min_size, max_size (float): the minimum and maximum size of the jump
        skip_num_points (int): The number of clean points to leave at the beginning
        
    RETURNS:
        y_glitch (array, float): the signal with glitches
    """
    
    # Randomly pick which points will be jumped
    num_glitches=np.random.choice(range(1,num_glitches_range+1))
    
    y_glitch=np.copy(y)
        
    randinds=np.random.randint(skip_num_points+1,len(y)-1, size=num_glitches)
    
    # Add jumps of random size, between min_size and max_size
    for randind in randinds:
        y_glitch[randind]+=(np.random.random()*(max_size-min_size)+min_size)*np.random.choice([-1,1])
    
    
    if return_inds:
        return y_glitch, randinds
    else:
        return y_glitch
    
    
def add_monochrom_glitch(y, 
                         num_glitches_range=2,
                         min_size=0.2,
                         max_size=0.5,
                         skip_num_points=10,
                         return_inds=False):
    """
    Adds a random number of monochromator glitches to the signal. A monochromator
    glitch is a positive point glitch immediately followed by a negative point
    glitch, or it is two or more successive point glitches.
    """
    
    # Randomly pick which points will be jumped
    num_glitches=np.random.choice(range(1,num_glitches_range+1))
    
    y_glitch=np.copy(y)
        
    randinds=np.random.randint(skip_num_points+1,len(y)-3, size=num_glitches)
    
    # Add jumps of random size, between min_size and max_size
    for randind in randinds:
        y_glitch[randind]+=(np.random.random()*(max_size-min_size)+min_size)
        y_glitch[randind+1]-=(np.random.random()*(max_size-min_size)+min_size)/2
    
    
    if return_inds:
        return y_glitch, randinds
    else:
        return y_glitch


def add_step_glitch_clean_start(y,
                                min_size=0.5,
                                max_size=1,
                                skip_num_points=10,
                                return_ind=False):
    """
    Adds a step glitch to the signal at a random location. 
    Same usage as add_point_glitch
    """
    
    ind = np.random.randint(skip_num_points+1,len(y)-1)
    
    y_glitch=np.copy(y)
    
    y_glitch[ind:]+=np.random.random()*(max_size-min_size)+min_size
    
    
    if return_ind:
        return y_glitch, ind
    else:
        return y_glitch
    
    
def Grubbs_test(r, sig_val=0.001):
    """
    Conducts Grubb's test with a given set of residuals r. 
    """
    
    mean=np.mean(r) # Calculate the mean of the residuals (ignoring NaN values)
    std=np.std(r) # Calculate the standard deviation
    max_ind=np.argmax(abs(r-mean)) # Find the index of the residual furthest from the mean

    test_stat=abs(r[max_ind]-mean)/std # Distance of the residual from the mean (of the residuals)
                                    # in units of standard deviations
    # Some intermediate values
    df=len(r)
    p=1-sig_val/df/2 
    t=stats.t.ppf(p, df-2)
    critical_val=(df-1)*t/np.sqrt((df-2+t*t)*df) # Maximum acceptable distance from the mean
                                                    # in units of standard deviations

    # Return the index of the outlier
    if test_stat>critical_val:
        return max_ind
    else:
        return None



def deglitch(glitchy,
             model=None,
             chunk_size=10,
             out_size=1,
             sig_val=0.025,
             return_all=False,
             noiseless=np.array([])):
    """
    Takes a glitchy signal, and deglitches it using the model provided
    """
    
    deglitched=glitchy.copy()
    num_points=len(glitchy)
    # Reset the hidden and cell states of the LSTM cell
    model.init_hidden(1)
    
    # Initialize lists to track the locations of the glitches
    point_glitches=[]
    step_glitches=[]
    predictions=[]
    
    # Create some first predictions without allowing it to report a glitch
    #for ind in range(chunk_size, 2*chunk_size):
    #    
    #    chunk=torch.Tensor([deglitched[ind-chunk_size:ind]])
    #    prediction=model(chunk).item()
    #    predictions.append(prediction)
    
    # FOR GRUBB'S TEST
    predictions=list(deglitched[:chunk_size]+np.random.normal(0, 0.02, chunk_size))
    
    # ind is the index of the value of the signal to be predicted
    # Cycle over every point after ind=chunk_size to look for and fix glitches
    for ind in range(chunk_size, num_points-out_size):
        
        # chunk: the current 10 values from the signal to make a prediction
        chunk=torch.Tensor(deglitched[ind-chunk_size:ind]).float().detach()
        # val: the value of the signal immediately after the last value in chunk
        val=deglitched[ind]
        
        # THRESHOLD METHOD
        # The threshold is 5 times the root mean squared error
        # of the predictions relative to the values for the last chunk
        #if ind<2*chunk_size:
        #    threshold=0.4
        #else:
        #    threshold=max(0.1,5*np.sqrt(np.mean((predictions[-chunk_size:]-chunk.squeeze().numpy())**2)))
        
        # prediction: the value after of the signal chunk predicted by the model
        # Check if we're using a model or the noiseless spectrum for predictions
        if len(noiseless)>0:
            prediction=noiseless[ind]
        else:
            prediction=model(chunk.view(1,1,-1)).squeeze().item()
        predictions.append(prediction)
        
        
        # GRUBB'S TEST METHOD
        r=np.array(predictions[-(chunk_size+1):])-np.append(chunk.squeeze().numpy(), val)
        # If there is an outlier, then Grubbs returns the index in the chunk
        # If deglitching has been working properly, then there should only ever
        # be a glitch in the last spot
        chunk_glitch_ind=Grubbs_test(r, sig_val)         


        # THRESHOLD METHOD
        # If the difference between the predicted value and the actual value
        # exceeds a threshold, it is probably a glitch/anomoly
        #if abs(prediction-val)>threshold and ind>chunk_size:
        
        # GRUBBS TEST METHOD
        # If the Grubb's test identified the previous value as an outlier
        # from the ten values before that, then it is a glitch
        if chunk_glitch_ind==chunk_size:
            
            # Try replacing the problem point with the predicted value
            next_ind=ind+1
            deglitched[ind]=prediction

            # Make a prediction for the next point 
            chunk2=torch.Tensor(deglitched[next_ind-chunk_size:next_ind]).float().detach()
            
            if len(noiseless)>0:
                prediction2=noiseless[next_ind]
            else:
                prediction2=model(chunk2.view(1,1,-1)).squeeze().item()
            
            val2=deglitched[next_ind]
            
            r=(np.append(predictions[-chunk_size:], prediction2)
            -np.append(chunk2.squeeze().numpy(), val2))
            chunk_glitch_ind=Grubbs_test(r, sig_val/4)
            
            # THRESHOLD METHOD
            # If the next point is still far from the prediction
            # It is probably a step glitch
            #if abs(prediction2-val2)>threshold:
            
            
            # GRUBB'S TEST METHOD
            if chunk_glitch_ind==chunk_size:
            
                # Subtract the inital step from the rest of the spectrum
                
                
                #print("Step glitch found!")
                # try this 
                deglitched[ind]=val
                #print(deglitched[ind])
                #print(prediction2, val2)
                
                #step=(prediction+prediction2-val-val2)/2
                step=prediction-val
                #print(step)
                deglitched[ind:]+=step
                

                
                #deglitched[next_ind:]+=(prediction-val)
                # Record the index of the glitch
                step_glitches.append(ind)
            
            # Then if the next point is close to the predicted value
            # then it was probably a point glitch and has been fixed
            else:
                # Record the index of the glitch
                point_glitches.append(ind)
                
                
    if return_all:
        return deglitched, predictions, point_glitches, step_glitches
    else:
        return deglitched
    
    




if __name__=="__main__":
     
    
    # Create places for the original signal, the one with added glithces, 
    # the deglitched signal, and the list of predicted points
    chi_dir="mu_test"
    chilist=os.listdir(chi_dir)
    chiname=np.random.choice(chilist)
    chi=np.genfromtxt(os.path.join(chi_dir, chiname))#[75:]
    x=np.arange(len(chi))
    #clean=chi/(max(chi)-min(chi))
    clean=chi.copy()
    
    
    noise_std=0.01
    noise=np.random.normal(0, noise_std, len(clean))
    glitchy=clean+noise
    
    add_glitches=True
    if add_glitches:
        glitchy, step_glitch_ind =add_step_glitch_clean_start(glitchy,
                                            min_size=0.1,
                                            max_size=0.12,
                                            skip_num_points=chunk_size,
                                            return_ind=True)
        glitchy, point_glitch_inds=add_point_glitch_clean_start(glitchy,
                                             min_size=0.1,
                                             max_size=0.12,
                                             skip_num_points=chunk_size,
                                             return_inds=True)
    
           
        
    # ========================================================================
    # The part to include in a separate project
    # In the next version, I'll put this all together in a class so that there
    # is a clean, one line of code that takes in a glitchy signal, and outputs
    # a deglitched signal, hiding the model-loading and the signal 
    # transformations that are going to be included.
    
    # Load the trained model
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    chunk_size=16
    hidden_size=32
    out_size=1
    sig_val=0.0005
    batch_size=1
    num_layers=4
    bidirectional=False
    drop_prob=0.5
    model=LSTMModel(chunk_size, 
                    hidden_size, 
                    out_size, 
                    batch_size,
                    num_layers, 
                    drop_prob,
                    bidirectional,
                    device).to(device)
    model.load_state_dict(torch.load("lstm_mu2.pth",
                                     map_location=torch.device(device)))
    model.init_hidden(batch_size)
    
    # Glitchy signal in, deglitched signal out
    (deglitched, 
     predictions,
     point_glitch_guesses,
     step_glitch_guesses) = deglitch(glitchy, 
                                    model,
                                    chunk_size,
                                    out_size,
                                    sig_val,
                                    True,
                                    noiseless=[])
    # =========================================================================



    # Visualize the results
    r=clean[:-out_size]-np.array(predictions)
    for i in point_glitch_guesses:  
        print("Possible point glitch at x=%.2f"%x[i])
       
    for i in step_glitch_guesses:
        print("Possible step glitch at x=%.2f"%x[i])
        
        
    # Plot the results
    plt.figure(1, figsize=(10,6))
    plt.plot(x, clean, label="Clean")
    plt.plot(x, glitchy, label="Glitchy")
    plt.plot(x, deglitched, label="Deglitched")
    plt.scatter(x[:-out_size], predictions,
                s=9, 
                marker="o", 
                color="r",
                label="Predictions")
    plt.scatter(x[1:], glitchy[:-1],
            s=9, 
            marker="o", 
            color="k",
            label="Glitchy lagged")
    plt.grid()
    plt.legend()
    plt.savefig("Output plot", dpi=200)
    
    plt.figure(2)
    plt.hist(r, bins=50)
    print("Mean squared error: %.4f"%(r**2).mean())
    print("Standard deviation of the error: %.4f"%r.std())
