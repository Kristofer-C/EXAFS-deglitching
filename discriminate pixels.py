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
    go from 0 to 1. This needs to be a separate function from find_good_pix
    because one of the pixels is always full of nans, and this is the place
    where that is checked and taken care of.
    """
    
    normY=[]
    
    for i, y in enumerate(Y):
        

        mn=min(y)
        mx=max(y)
        if mn==mx or mx!=mx:
            y=np.zeros_like(y)
        else:
            y=(y-mn)/(mx-mn)
        
        normY.append(y)  
            
    return np.stack(normY)


def find_good_pix(mus, min_samples=4, eps=0.05):    
    """
    Sorts the normalized pixels scans by clustering the mean and standard 
    deviation of the consecutive differences of each. Returns the list of
    good pixel scans.
    """
    
    normmus=norm_transform(mus)
    diffs=normmus[:,1:]-normmus[:,:-1]
    means=diffs.mean(axis=1)
    normmns=(means-min(means))/(max(means)-min(means))
    stds=diffs.std(axis=1)
    normstds=(stds-min(stds))/(max(stds)-min(stds))
    
    X=np.stack([normstds, normmns], axis=1)
    db=DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    Dlabels=db.labels_
    
    # Visualization for debugging
    #plt.figure(4)
    #plt.scatter(normmns[Dlabels!=-1], normstds[Dlabels!=-1], color='b', alpha=0.5)
    #plt.scatter(normmns[Dlabels==-1], normstds[Dlabels==-1], color='k', alpha=0.5)
    
    return np.arange(len(mus))[Dlabels!=-1]



def similarity_score(Y):
    """
    The average mean squared error between the signals in the group Y and the
    average of the signals in group Y.
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
    
    # Run the three functions
    normmus=norm_transform(mus)
    good_pix_inds=find_good_pix(mus)
    display_difference(normmus, good_pix_inds)
    
