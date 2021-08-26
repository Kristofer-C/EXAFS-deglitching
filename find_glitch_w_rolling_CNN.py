# -*- coding: utf-8 -*-
"""
Uses a model to identify whether a chunk contains a step or monochromator glitch
anywhere in it, and reports the range of x values for each glitch.

Updated: reports just the last point in the range. It is usually the one closest
to the real glitch.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from Rolling_CNN_classifier import RollingCNN
from scipy.ndimage import gaussian_filter

# A function for bunching up a sequence
# One day hopefully I'll find a pytorch method to do it without a for loop.
def chunk_sequence(sequence, chunk_size=10, n_leave=0):
    
    chunks=[]
    for i in range(0,len(sequence)-chunk_size+1-n_leave):
        chunks.append(sequence[i:i+chunk_size])
        
    return torch.stack(chunks)

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
        y_glitch[randind]+=(np.random.random()*(max_size-min_size)+min_size)*\
                                                        np.random.choice([-1,1])
    
    
    if return_inds:
        return y_glitch, randinds
    else:
        return y_glitch
    
    
def add_mcglitch(y,
                 glitch,
                 num_glitches_range=2,
                 min_size=0.2,
                 max_size=0.5,
                 skip_num_points=10,
                 return_inds=False):
    """
    Adds a random number of monochromator glitches to the signal.
    """
    
    # Randomly pick which points will be jumped
    num_glitches=np.random.choice(range(1,num_glitches_range+1))
    
    y_glitch=np.copy(y)
        
    randinds=np.random.randint(skip_num_points+1,len(y)-len(glitch), size=num_glitches)
    
    # Add jumps of random size, between min_size and max_size
    for randind in randinds:
        y_glitch[randind:randind+len(glitch)]+=glitch*(np.random.random()*\
                        (max_size-min_size)+min_size)*np.random.choice([-1,1])    
    
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
    
    step=(np.random.random()*(max_size-min_size)+min_size)*np.random.choice([-1,1])
    # Sometimes a step glitch takes two points
    y_glitch[ind]+=np.random.choice([0,1])*step/2
    y_glitch[ind+1:]+=step
    
    
    if return_ind:
        return y_glitch, ind
    else:
        return y_glitch
    
    
    



def find_glitchy_inds(glitchy,
                      f_model,
                      chunk_size=10):
    
    """
    Scans the signal with the glitch classification model to find the indices
    of the beginning of a chunk that contains a step or monochromator glitch.
    """
    
    x=torch.arange(len(glitchy)-chunk_size+1)
    
    chunks=chunk_sequence(glitchy, chunk_size).float()
    mn=chunks.min(dim=1)[0].view(-1,1)
    mx=chunks.max(dim=1)[0].view(-1,1)
    chunks=(chunks-mn)/(mx-mn)
    #chunks-=chunks.mean(dim=1).view(-1,1)
    p=f_model(chunks)
    argsmax=torch.argmax(p, dim=1)
    
    # Chunks are indexed by their first point
    # Lists of indices of chunks containing monochromator (mc) glitches and 
    # step (s) glitches respectively
    mcglitch_inds=x[argsmax==1]
    sglitch_inds=x[argsmax==2]
    
            
    return mcglitch_inds, sglitch_inds
            
        
        
def ind_ranges(inds):
    """
    Takes a list of indices and returns a list of pairs of the starting and ending
    points of consecutive indices.
    """
    
    start_inds=[] # List of indices that begin a segment of consecutive indices
    end_inds=[] # List of indices that end a segment of consecutive indices
    
    if len(inds)==0: # If inds is an empty list, don't try to find beginning
        # and end points
        return list(zip(start_inds, end_inds))
        
    # The first item is always a starting point
    start_inds.append(inds[0])
        
    # Cycle through items. If consecutive items are not consecutive indices,
    # then there is a split between two segments
    for i in range(0, len(inds)-1):
        if inds[i]+1!=inds[i+1]:
            end_inds.append(inds[i])
            start_inds.append(inds[i+1])
    
    # The last item is always an ending point
    end_inds.append(inds[-1])
            
    # return the pairs of starting and ending points
    return list(zip(start_inds, end_inds))


def get_glitch_locs(mcglitch_guesses,
                    sglitch_guesses, 
                    min_mc_range, 
                    min_s_range):
    
    """
    Takes the list of guesses for both types of glitch and returns the 
    approximate locations of the guesses, accounting for the minimum distance
    between them and the minimum range over which the algorithm detected the 
    same glitch.
    """
    
    mcglitch_ranges=np.array(ind_ranges(mcglitch_guesses))
    sglitch_ranges=np.array(ind_ranges(sglitch_guesses))
    
    if mcglitch_ranges.size!=0:
        mcglitch_locs=\
        mcglitch_ranges[(mcglitch_ranges[:,1]-mcglitch_ranges[:,0]+1)>=min_mc_range, 1]
    else:   
        mcglitch_locs=[]
    
    if sglitch_ranges.size!=0:
        sglitch_locs=\
        sglitch_ranges[(sglitch_ranges[:,1]-sglitch_ranges[:,0]+1)>=min_s_range, 1]
    else:
        sglitch_locs=[]


    return mcglitch_locs, sglitch_locs
    

    
def run_glitch_finding():
    """
    I just put the main script into the function so that I can run a different
    thing designed for tesing, debugging, and characterizing
    """
    
    # Load the trained model
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device='cpu'
    print("Device: %s"%device)
    out_size=3
    find_model=RollingCNN(out_size).to(device)
    find_model.load_state_dict(torch.load("Rolling_glitch_classifier1.pth", map_location=torch.device(device)))

    test_dir="mu_test"
    chunk_size=16

    
    # The minimum length of the range of the detected glitch for it to be 
    # considered a probable glitch
    min_mc_range=2
    min_s_range=4
    
        
    # Create places for the original signal, the one with added glithces, 
    # the deglitched signal, and the list of predicted points
    # Pick a random signal from the specified directory
    fclean=np.random.choice(os.listdir(test_dir))
    clean=np.genfromtxt(os.path.join(test_dir, fclean))
    x=np.arange(len(clean))
    # Add extra noise
    noisy=np.random.normal(0,0.01,len(x))+clean
    
    # Right now, show the results of the algorithm on both the glitchy and
    # normal spectra
    # Otherwise, add_glitches acts as a convenient toggling parameter testing
    for add_glitches in [False, True]:
        glitchy=noisy.copy()
        if add_glitches:
            
            # Add a step glitch            
            glitchy, sglitch_ind =add_step_glitch_clean_start(glitchy,
                                                min_size=0.1,
                                                max_size=0.15,
                                                skip_num_points=chunk_size,
                                                return_ind=True)
            
            # For end-of-chunk detection, use point glitches
            glitchy, mcglitch_inds=add_point_glitch_clean_start(glitchy,
                                                 min_size=0.12,
                                                 max_size=0.15,
                                                 skip_num_points=chunk_size,
                                                 return_inds=True)
            
            # For middle-of-chunk classification, use monocromator glitches
            #mcglitch=np.genfromtxt("monochromator_glitch.txt")
            # No point using a file for 5 points. We might want this to look 
            # different in the future though.
            mcglitch=np.array([-0.06666667, 
                               -0.33333333, 
                               -1.00000000, 
                               0.33333333, 
                               0.11111111])
            #glitchy, mcglitch_inds=add_mcglitch(glitchy,
            #                                 mcglitch,
            #                                 num_glitches_range=2,
            #                                 min_size=0.08,
            #                                 max_size=0.09,
            #                                 skip_num_points=chunk_size,
            #                                 return_inds=True)

        # Normalize the signal
        glitchy=(glitchy-min(glitchy))/(max(glitchy)-min(glitchy))
        t0=time.time()
        # Get the locations at which the model found a glitch
        mcglitch_guesses, sglitch_guesses= \
        find_glitchy_inds(torch.Tensor(glitchy).to(device), find_model, chunk_size)
        
        # Get the ends of the ranges for more specific locations
        mcglitch_locs, sglitch_locs = get_glitch_locs(mcglitch_guesses,
                                                      sglitch_guesses,
                                                      min_mc_range,
                                                      min_s_range)
        print("That took: %.3f seconds"%(time.time()-t0))
    
        # Print thelocations of the glitches found. 
        for loc in mcglitch_locs:  
            print("Possible monochromator glitch near x=%.2f"\
                  %x[loc])
            
           
        for loc in sglitch_locs:
            print("Possible step glitch near x=%.2f"\
                  %x[loc])
        
        if add_glitches:
            # For reference, the locations of the actual glitches
            print("True monochromator glitches at: x=", x[mcglitch_inds])
            print("True step glitch at x=%.2f"%x[sglitch_ind])
            
            
        # Plot the results  
        
        plt.figure(figsize=(10,6))
        plt.plot(x, clean, label="Clean", color='g')
        plt.plot(x, glitchy, label="Glitchy")
        if len(mcglitch_locs)!=0:
            plt.scatter(x[mcglitch_locs],
                        glitchy[mcglitch_locs],
                        color='r', 
                        label="Pred. monochromator glitches",
                        s=64)
        if len(sglitch_locs)!=0:
            plt.scatter(x[sglitch_locs], 
                        glitchy[sglitch_locs],
                        color='k', 
                        label="Pred. step glitch",
                        s=64)
        
    
        
        plt.legend()
    #plt.savefig("Output plot", dpi=200)
    print(mcglitch_guesses)
    print(sglitch_guesses)
    

def run_testing():
    """A different way to run the whole thing that allows for more high-volume
    testing of different parameters."""
    
    # Load the trained model
    #device="cpu"
    test_dir="mu_test"
    chunk_size=16
    out_size=3
    min_range=6
    
    find_model=RollingCNN(out_size)#.to(device)
    find_model.load_state_dict(torch.load("Rolling_glitch_classifier1.pth"))
    
    # the ranges of indeces for the glitch guesses where the algorithm detected
    # them correctly, and the ranges for when they were false positives
    pr=[]
    nr=[]
    dists=[]
    #tp=0
    #fp=0
    
    for i in range(100):
        # Create places for the original signal, the one with added glithces, 
        # the deglitched signal, and the list of predicted points
        fclean=np.random.choice(os.listdir(test_dir))
        clean=np.genfromtxt(os.path.join(test_dir, fclean))
        x=np.arange(len(clean))
        # Option to add extra noise
        clean+=np.random.normal(0, 0.02, len(x))
        
        
        glitchy=clean.copy()
        add_glitches=True
        if add_glitches:
            # Add a step glitch
            glitchy, sglitch_ind =add_step_glitch_clean_start(glitchy,
                                                min_size=0.2,
                                                max_size=0.25,
                                                skip_num_points=chunk_size,
                                                return_ind=True)
            
            # For end-of-chunk detection, use point glitches
            #glitchy, point_glitch_inds=add_point_glitch_clean_start(glitchy,
            #                                     min_size=0.3,
            #                                     max_size=0.4,
            #                                     skip_num_points=chunk_size,
            #                                     return_inds=True)
            
            # For middle-of-chunk classification, use monocromatory glitches
            #mcglitch=np.genfromtxt("monochromator_glitch.txt")
            # No point using a file for 5 points. We might want this to look different
            # in the future though.
            mcglitch=np.array([-0.06666667, 
                               -0.33333333, 
                               -1.00000000, 
                               0.33333333, 
                               0.11111111])
            glitchy, mcglitch_inds=add_mcglitch(glitchy,
                                             mcglitch,
                                             num_glitches_range=2,
                                             min_size=0.2,
                                             max_size=0.25,
                                             skip_num_points=chunk_size,
                                             return_inds=True)
        
       
        # Only take a glitch detection seriously if it was found more than once
        # and if the step glitch was found far from a monochromator glitch
        mcglitch_guesses, sglitch_guesses= \
                                    find_glitches(glitchy, find_model, chunk_size)
        mcglitch_ranges=np.array(ind_ranges(mcglitch_guesses))
        sglitch_ranges=[]
        if mcglitch_ranges.size!=0:
            mcglitch_ranges\
                    =mcglitch_ranges[(mcglitch_ranges[:,1]-mcglitch_ranges[:,0])>min_range]
    
        
        for pair in ind_ranges(sglitch_guesses):
            if (pair[1]-pair[0])>min_range:
                if mcglitch_ranges.size==0:
                    sglitch_ranges.append(pair)
                elif (abs(pair[0]-mcglitch_ranges[:,1])>1).all():
                    sglitch_ranges.append(pair)
    
        sglitch_ranges=np.array(sglitch_ranges)
        
        if sglitch_ranges.size!=0:
            dists.extend(sglitch_ranges[:,1]-sglitch_ind)
        if mcglitch_ranges.size!=0:
            for pair in mcglitch_ranges:
                d=min(pair[1]-mcglitch_inds)
                dists.append(d)
        
        for pair in mcglitch_ranges:
            if (abs(pair[1]-mcglitch_inds)<=1).any():
                pr.append(pair[1]-pair[0])
            else:
                nr.append(pair[1]-pair[0])
           #     tp+=1
           # else:
           #     fp+=1
       # 
        for pair in sglitch_ranges:
            if abs(pair[1]-sglitch_ind)<=1:
                pr.append(pair[1]-pair[0])
            else:
                nr.append(pair[1]-pair[0])
         #       tp+=1
         #   else:
         #       fp+=1
                
    plt.hist([np.array(pr), np.array(nr)], bins=40)
    #print(tp, fp)
    #return dists

    
if __name__=="__main__":
    
    run_glitch_finding()