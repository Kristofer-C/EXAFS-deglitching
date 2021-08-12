# -*- coding: utf-8 -*-
"""
Identifies the pixels worth keeping in a fluorescence EXAFS scan.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def norm_transform(Y):
    """
    Transforms a set of signals Y into such that each signal is rescaled to 
    go from 0 to 1. 
    """
    
    # The list of normalized signals
    normY=[]
    
    for y in Y:
        
        mn=min(y)
        mx=max(y)
        
        # If the signal is a flat line or the maximum is nan,
        # just set the signal to be all zeros.
        if mn==mx or mx!=mx:
            y=np.zeros_like(y)
        else:
            # If the signal has information in it, rescale it to
            # go from 0 to 1.
            y=(y-mn)/(mx-mn)
        
        normY.append(y)  
            
    return np.stack(normY)


def find_good_pix(mus, min_samples=4, eps=0.05):    
    """
    Sorts the normalized pixels scans by clustering the mean and standard 
    deviation of the consecutive differences of each. Returns the indeces of
    good pixel scans.
    """
    
    # Normalize all the pixel scans, leaving the flat lines and nans as zeros
    normmus=norm_transform(mus)
    
    # Get the lists of consecutive differences, and calculate the means
    # and standard deviations.
    # With the lists of means and standard deviations, rescale them
    # such that the lowest is zero and the highest is one.
    diffs=normmus[:,1:]-normmus[:,:-1]
    means=diffs.mean(axis=1)
    normmns=(means-min(means))/(max(means)-min(means))
    stds=diffs.std(axis=1)
    normstds=(stds-min(stds))/(max(stds)-min(stds))
    
    # Perform DBSCAN clustering on the scans in the
    # means/standard deviations space.
    X=np.stack([normstds, normmns], axis=1)
    db=DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    
    # The list of labels for each scan describing which 
    # cluster it beloings to. A label of -1 is no cluster.
    # Clusters are numbered starting from zero.
    Dlabels=db.labels_
    
    # Visualization of the clustering for debugging
    #plt.figure(4)
    #plt.scatter(normmns[Dlabels!=-1], normstds[Dlabels!=-1], color='b', alpha=0.5)
    #plt.scatter(normmns[Dlabels==-1], normstds[Dlabels==-1], color='k', alpha=0.5)
    
    # Return the indeces of the scans that belong to a cluster.
    return np.arange(len(mus))[Dlabels!=-1]



def similarity_score(Y):
    """
    The average mean squared error between the signals in the group Y and the
    average of the signals in group Y. The similarity score describes how
    Similar the clustered scans are to each other.
    """
    
    return sum(sum((Y-Y.mean(axis=0))**2))/len(Y)
    


def display_difference(mus, good_pix_inds):
    """
    Displays the effect of the discrimination process by plotting the kept and
    discrinimated scans in different colours, and showing the summation with
    and without discrimination. The true final sum will look different because
    the pixels will have different weights.
    """
    
    
    plt.figure(1, figsize=(8,6))
    plt.title("Discriminated normalized pixel scans")
    line,=plt.plot([],[],color='b', label='%d good pixels'%len(good_pix_inds))
    line,=plt.plot([],[],color='k', label='%d discriminated pixels'%(len(mus)-len(good_pix_inds)))
    plt.legend()
    
    for i, mu in enumerate(mus):
        
        if i in good_pix_inds:
            color='b'
        else:
            color='k'
        plt.plot(mu, color=color)
        
    plt.figure(2, figsize=(8,6))
    plt.plot(np.sum(mus[good_pix_inds], axis=0)/len(good_pix_inds), color='b', label='Sum with discrimination')
    plt.plot(np.sum(mus, axis=0)/(len(mus)), color='k', label='Sum without discrimination')   
    plt.legend()
    
    print("Similarity score without discrimination: %.4f"%similarity_score(mus))
    print("Similarity score with discrimination: %.4f"%similarity_score(mus[good_pix_inds]))
    
    
if __name__=="__main__":
    
    # Just an example I have on file
    MU=np.load("pixel_mus.npy", allow_pickle=True)
    E=np.load("pixel_es.npy", allow_pickle=True)
    NPIX=32
    start=np.random.choice(range(64))
    print("Start: %d"%start)
    es=E[start*NPIX:(start+1)*NPIX]
    mus=np.stack(MU[start*NPIX:(start+1)*NPIX])
    
    
    # This function is mandatory
    # Get the indeces of the clustered scans
    good_pix_inds=find_good_pix(mus)
   
    # Optional: Display the results.
    display_difference(mus, good_pix_inds)
    
