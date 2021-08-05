# Identifies and removes point and step glithes in mu(e)
# Uses the chi extractor from AcquamanHDFExplorer to transform mu into chi,
# perform the deglitching on chi, and then transform back
# Still very much a work in progress but the form is there

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.interpolate import LSQUnivariateSpline

from LSTM_on_chi1 import LSTMModel
from discriminate_pixels import find_good_pix

ch = 12398.419297617678  # c*h[eV*A]
eV2revA = 0.2624682843  # 2m_e(eV)/(hbar(eVs)c(Å/s))^2


# Taken with small changes from AcquamanHDFExplorer
# Modified only to cut it safely from the original script
# The reverse function was added
class Mu_to_Chi_Transform():
    
    def __init__(self):
        super(Mu_to_Chi_Transform, self).__init__()
        self.pre1 = -150
        self.pre2 = -50
        self.post1 = 50
        self.r = np.arange(0.0, 6.0, 0.02)
        self.hintE0 = []


    def victoreenMod(self, e, c, d):
        f = ch / e
        return c*f**-3 + d*f

    def e0(self, e, mu=None):
        try:
            testMu = mu
            testE = e
            indD = 0
            deriv = np.gradient(testMu) / np.gradient(testE)
            deriv = gaussian_filter1d(deriv, 3)
            ind = np.argmax(deriv)
            return ind + indD, testE[ind], deriv[ind]
        except (IndexError, ValueError):
            return
    
    def pre_edge(self, e, ie0, mu):
        try:
            idx1, idx2 = \
                np.searchsorted(e, [e[ie0]+self.pre1, e[ie0]+self.pre2])
            popt, pcov = \
                curve_fit(self.victoreenMod, e[idx1:idx2+1], mu[idx1:idx2+1])
            return self.victoreenMod(e, popt[0], popt[1])
        except TypeError:
            return
    
    def k(self, e, mu):
        rese0 = self.e0(e, mu)
        if rese0 is None:
            print("rese0 is None.")
            return
        ie0 = rese0[0]
        k = np.sqrt(eV2revA * (e[ie0:]-e[ie0]))
        kmin_idx = np.searchsorted(e, e[ie0]+self.post1) - ie0
        if kmin_idx > len(k) - 2:
            print("kmin_idx problem.")
            return
        return ie0, k, kmin_idx
    
    def chi(self, e, ie0, mu, pre, k, kmin_idx, kw=2):
        # k starts from e0
        fun_fit = (mu[ie0:] - pre[ie0:]) * k**kw
        n_knots = int(2.0 * (k.max()-k.min()) / np.pi) + 1
        knots = np.linspace(k[kmin_idx], k[-2], n_knots)
    #        print(k, fun_fit, knots)
    #        print(np.diff(k))
        spl = LSQUnivariateSpline(k, fun_fit, knots)
        self.post = spl(k[kmin_idx:]) / k[kmin_idx:]**kw
        self.edge_step = self.post[0]
        chi = (mu - pre)[ie0:] - self.edge_step
        chi[kmin_idx:] += self.edge_step - self.post
        chi /= self.edge_step
        
        return chi
        
        
    def forward(self, e, mu):

        xax = np.copy(e)
        diffTest = np.diff(xax) <= 0
        if any(diffTest):
            xax[np.where(diffTest)] -= 1e-1

        try:

            kres = self.k(xax, mu)
            if kres is None:
                print("kres is None.")
                return
            self.e0_idx, k, self.kmin_idx = kres

            self.pre = self.pre_edge(xax, self.e0_idx, mu)
            if self.pre is None:
                print("pre_edge is None")
                return
            chi = self.chi(xax, self.e0_idx, mu, self.pre, k, self.kmin_idx)
            self.scale=max(chi)-min(chi)
            chi/=self.scale
        except ValueError:
            print("Value error.")
            return

        return k, chi


    def reverse(self, chi, mu):
        """
        Transform chi back to mu using the parameters used to transform
        mu to chi.
        """
        
        y=np.copy(chi)
        y*=self.scale
        y*=self.edge_step
        y[self.kmin_idx:]-=self.edge_step-self.post
        y+=self.edge_step+self.pre[self.e0_idx:]
        
        y=np.concatenate([mu[:self.e0_idx], y])
        
        return y
        


class Mu_Deglitcher():
    
    def __init__(self):
        
        # Load the model
        device='cpu'        
        hidden_size=32        
        batch_size=1
        num_layers=4
        bidirectional=False
        drop_prob=0.5
        
        self.chunk_size=16
        self.out_size=1
        self.model=LSTMModel(self.chunk_size, 
                             hidden_size, 
                             self.out_size, 
                             batch_size,
                             num_layers, 
                             drop_prob,
                             bidirectional,
                             device).to(device)
        self.model.load_state_dict(torch.load("lstm_chi1.pth",
                                              map_location=torch.device(device)))
        self.model.init_hidden(batch_size)
        
        # Define the transform class
        self.transform=Mu_to_Chi_Transform()
        

    def Grubbs_test(self, r, sig_val):
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
        
        

    def deglitch(self,
                 glitchy,
                 sig_val, 
                 return_all):
        """
        Takes a glitchy signal, and deglitches it using the model provided
        """
        
        deglitched=glitchy.copy()
        num_points=len(glitchy)
        # Reset the hidden and cell states of the LSTM cell
        self.model.init_hidden(1)
        
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
        # Grubb's test requires a short history of residuals. We obviously can't
        # make predictions before the chunk_size+1th point, so the predictions 
        # are initialized by adding noise to the actual signal
        predictions=list(deglitched[:self.chunk_size]+np.random.normal(0, 0.02, self.chunk_size))
        
        # ind is the index of the value of the signal to be predicted
        # Cycle over every point after ind=chunk_size to look for and fix glitches
        # Don't make a prediction for the last point, becasue a possible step
        # glitch requires there to be another point for comparison
        for ind in range(self.chunk_size, num_points-self.out_size):
            
            # chunk: the current chunk_size values from the signal that are used
            # to make a prediction
            chunk=torch.Tensor(deglitched[ind-self.chunk_size:ind]).float().detach()
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
            prediction=self.model(chunk.view(1,1,-1)).squeeze().item()
            predictions.append(prediction)
            
            
            # GRUBB'S TEST METHOD
            # r: the residuals from between last chunk_size predictions and the 
            # given signal
            r=np.array(predictions[-(self.chunk_size+1):])-np.append(chunk.squeeze().numpy(), val)
            # If there is an outlier, then Grubbs returns the index in the chunk
            # If deglitching has been working properly, then there should only ever
            # be a glitch in the last spot
            chunk_glitch_ind=self.Grubbs_test(r, sig_val)         
    
    
            # THRESHOLD METHOD
            # If the difference between the predicted value and the actual value
            # exceeds a threshold, it is probably a glitch/anomoly
            #if abs(prediction-val)>threshold and ind>chunk_size:
            
            # GRUBBS TEST METHOD
            # If the Grubb's test identified the previous value as an outlier
            # from the chunk_size values before that, then it is a glitch
            if chunk_glitch_ind==self.chunk_size:
                
                # Try replacing the problem point with the predicted value
                next_ind=ind+1
                deglitched[ind]=prediction
    
                # Make a prediction for the next point 
                chunk2=torch.Tensor(deglitched[next_ind-self.chunk_size:next_ind]).float().detach()
                prediction2=self.model(chunk2.view(1,1,-1)).squeeze().item()
                val2=deglitched[next_ind]
                
                r=(np.append(predictions[-self.chunk_size:], prediction2)
                -np.append(chunk2.squeeze().numpy(), val2))
                
                chunk_glitch_ind=self.Grubbs_test(r, sig_val/4)
                
                # THRESHOLD METHOD
                # If the next point is still far from the prediction
                # It is probably a step glitch
                #if abs(prediction2-val2)>threshold:
                
                
                # GRUBB'S TEST METHOD
                if chunk_glitch_ind==self.chunk_size:
                
                    # Subtract the inital step from the rest of the spectrum
                    # Another method was tried and commented out

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
        
        
    
    
    def run(self, e, glitchy_mu, sig_val, return_all):
        
        k, chi=self.transform.forward(e, glitchy_mu)
        
        if return_all:
            
            # The guessed glitchy points are the indices of k, not e.
            (deglitched_chi,
            predictions,
            point_glitches,
            step_glitches)=self.deglitch(chi, sig_val, return_all)
            
            # Some visualization
            #plt.figure(3)
            #plt.plot(k, chi)
            #plt.scatter(k[:-self.out_size], predictions, s=8, color='r')
            #print("Possible point glitches at: ", k[point_glitches])
            #print("Possible step glitches at: ", k[step_glitches])
            
            deglitched_mu=self.transform.reverse(deglitched_chi, glitchy_mu)
            
            return deglitched_mu, predictions, point_glitches, step_glitches
            
            
        else:
            deglitched_chi=self.deglitch(chi, sig_val, return_all)
            deglitched_mu=self.transform.reverse(deglitched_chi, glitchy_mu)
            
            return deglitched_mu
        
        


# Functions for this script, not really useful for anything else:
            
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
    





if __name__=="__main__":
    
    # Load an example mu and e
    # Just an example I have on file
    MU=np.load("pixel_mus.npy", allow_pickle=True)
    E=np.load("pixel_es.npy", allow_pickle=True)
    NPIX=32
    #start=np.random.choice(range(64))
    start=58
    print("Start: %d"%start)
    es=E[start*NPIX:(start+1)*NPIX]
    mus=np.stack(MU[start*NPIX:(start+1)*NPIX])
    good_pix_inds=find_good_pix(mus)
    mu=np.sum(mus[good_pix_inds], axis=0)
    e=np.sum(es[good_pix_inds], axis=0)/len(good_pix_inds)
    
    glitchy=np.copy(mu)
    
    # Add glitches if desired
    add_glitches=False
    if add_glitches:
        glitchy, step_glitch_ind =add_step_glitch_clean_start(glitchy,
                                            min_size=max(glitchy)/20,
                                            max_size=max(glitchy)/15,
                                            skip_num_points=chunk_size,
                                            return_ind=True)
        glitchy, point_glitch_inds=add_point_glitch_clean_start(glitchy,
                                            min_size=max(glitchy)/20,
                                            max_size=max(glitchy)/15,                                             skip_num_points=chunk_size,
                                            return_inds=True)
    
    
    plt.figure(1)
    plt.plot(e,glitchy)

    # Here's the syntax for the use of the deglitcher
    Deglitcher=Mu_Deglitcher()
    #(deglitched_mu, 
    #predictions,
    #point_glitch_guesses,
    #step_glitch_guesses) = Deglitcher.run(e, mu, sig_val=0.005, return_all=True)
    
    # Or:
    deglitched_mu = Deglitcher.run(e, glitchy, sig_val=0.001, return_all=False)

    plt.figure(1)
    plt.plot(e, deglitched_mu)
    
