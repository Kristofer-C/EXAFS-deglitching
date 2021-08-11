# Identifies and removes point and step glithes in mu(E)
# Uses the chi extractor from AcquamanHDFExplorer to transform mu into chi,
# perform the deglitching on chi, and then transform back
# Still very much a work in progress but the form is there

import os
import time
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
    
    def chi(self, e, ie0, mu, pre, k, kmin_idx, kw=2, freeze_params=False):
        # k starts from e0
        fun_fit = (mu[ie0:] - pre[ie0:]) * k**kw
        n_knots = int(2.0 * (k.max()-k.min()) / np.pi) + 1
        knots = np.linspace(k[kmin_idx], k[-2], n_knots)
    #        print(k, fun_fit, knots)
    #        print(np.diff(k))
        spl = LSQUnivariateSpline(k, fun_fit, knots)
        
        if not freeze_params:
            self.post = spl(k[kmin_idx:]) / k[kmin_idx:]**kw
            self.edge_step = self.post[0]
        chi = (mu - pre)[ie0:] - self.edge_step
        chi[kmin_idx:] += self.edge_step - self.post
        chi /= self.edge_step
        
        if not freeze_params:
            self.scale=max(chi)-min(chi)
        chi/=self.scale
        
        return chi
        
        
    def forward(self, e, mu, freeze_params=False):

        xax = np.copy(e)
        diffTest = np.diff(xax) <= 0
        if any(diffTest):
            xax[np.where(diffTest)] -= 1e-1

        try:

            kres = self.k(xax, mu)
            if kres is None:
                print("kres is None.")
                return
            
            ie0, k, kmin_idx = kres
            if not freeze_params:
                self.e0_idx, self.kmin_idx = ie0, kmin_idx
                self.pre = self.pre_edge(xax, self.e0_idx, mu)
                
            if self.pre is None:
                print("pre_edge is None")
                return
        
            chi = self.chi(xax, self.e0_idx, mu, self.pre, k, self.kmin_idx,
                           freeze_params=freeze_params)

        except ValueError:
            print("Value error.")
            return

        return k, chi


    def reverse(self, chi, mu):
        """
        Transform a modified chi back to mu using the parameters used to 
        transform mu to chi. Mu is required in order to stitch the pre-edge
        mu to the transformed chi.
        """
        
        y=np.copy(chi)
        y*=self.scale
        y*=self.edge_step
        y[self.kmin_idx:]-=self.edge_step-self.post
        y+=self.edge_step+self.pre[self.e0_idx:]
        
        y=np.concatenate([mu[:self.e0_idx], y])
        
        return y
        


class Mu_Deglitcher():
    """
    Wraps all the functions necessary to perform the deglitching on mu.
    
    Variables:
    device ('cpu' or 'cuda'): The device to be used when running the model 
    on the chunks.
    chunk_size (int): The number of points in each chunk. Default is 16. 
    For the model currently being loaded, chunk_size must be 16.
    out_size (int): The number of floats outputted by the model. Default
    is 1. For the current model, out_size must be 1.
    model (pytorch module): The LSTM model used for predicting the next
    point in the sequence.
    Transform (class): The class used to transform mu to chi and back again.
    Default is Mu_to_Chi_transform(). It must contain .forward() and .reverse()
    which return the respective transformations. .forward() must have the 
    keyword boolean argument 'freeze_params'.
    
    Functions:
    Grubbs_test: Performs Grubb's test on an array
    deglitch: Removes sharp points and sudden steps from an array.
    run: Performs the transformations and deglitching.
    run_twostages: Performs the transformations and deglitching
    with a two-stage process.
    """
    
    def __init__(self):
        
        # Load the LSTM model with the required parameters.
        hidden_size=32        
        batch_size=1
        num_layers=4
        bidirectional=False
        drop_prob=0.5
        
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.chunk_size=16
        self.out_size=1
        self.model=LSTMModel(self.chunk_size, 
                             hidden_size, 
                             self.out_size, 
                             batch_size,
                             num_layers, 
                             drop_prob,
                             bidirectional,
                             self.device).to(self.device)
        self.model.load_state_dict(torch.load("lstm_chi1.pth",
                                              map_location=torch.device(self.device)))
        self.model.init_hidden(batch_size)
        
        # Define the transform class
        self.transform=Mu_to_Chi_Transform()
        

    def Grubbs_test(self, r, sig_val=0.025):
        """
        Conducts Grubb's test with a given set of residuals r. Returns
        the index of the item in r that is an outlier (if one exists)
        according to Grubb's test.
        
        Parameters:
        r (array, float): A 1D array of values, one of which may be an outlier.
        sig_val (float, default=0.025): The significance value with which to 
        identify the possible outlier according to Grubb's test. A higher
        sig_val is more likely to result in an identified outlier. Values 
        should be within an order of magnitude of the default for best results.
        
        Returns:
        max_ind (int): The index of the item in r that is identified as an outlier.
        If no outlier is identified, returns None.
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
                 sig_val=0.025, 
                 return_all=False):
        """
        Takes a glitchy signal, and deglitches it using the model provided. 
        
        A trained LSTM model is used to take sets of consecutive values from
        a signal and predict the next value in the sequence. The difference between
        the predicted value and the next measured value is compared to the most recent
        set of differences in predictions and measured values. If the latest difference
        between the predicted value and the measured value is determined to be an outlier
        by Grubb's test, then that point is labelled as a glitch. The measured value is 
        replaced by the predicted value. If the following measured value is determined to 
        be an outlier in the same way, the glitch is labelled as a step glitch and the
        rest of the signal is shifted by the difference between the predicted value and 
        the measured value. If the following point is not an outlier, then it was a point
        glitch and was already replaced by the corresponding predicted value. 
        
        Instead of Grubb's test, a threshold method can be used to identify outliers.
        The threshold is the maximum absolute difference a measured value can have from
        the predicted value for it to be labelled as normal data. The threshold is
        calculated as five times the root mean squared error between the most recent
        set of predicted values and their measured values.
        
        Parameters:
        glitchy (array, float): The set of values (signal) that is to be deglitched.
        sig_val (float, default=0.025): The significance value with which Grubb's test is 
        conducted. See Grubbs_test() for details.
        return_all (bool, default=False): If false, deglitched is returned.
        If true, deglitched, predictions, point_glitches, and step_glitches are returned.
        
        Returns:
        deglitched (float, array): The deglitched set of values.
        predictions (float, list): The list of predictions made by the model during
        deglitching.
        point_glitches (int, list): The list of indeces determined to contain point
        glitches
        step_glitches (int, list): The list of indeces determined to contain step
        glitches
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
        predictions=list(deglitched[:self.chunk_size]+np.random.normal(0, 0.04, self.chunk_size))
        
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
            prediction=self.model(chunk.view(1,1,-1).to(self.device)).squeeze().item()
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
                prediction2=self.model(chunk2.view(1,1,-1).to(self.device)).squeeze().item()
                val2=deglitched[next_ind]
                
                r=(np.append(predictions[-self.chunk_size:], prediction2)
                -np.append(chunk2.squeeze().numpy(), val2))
                
                chunk_glitch_ind=self.Grubbs_test(r, sig_val/2)
                
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
        
        
    
    
    def run(self, e, glitchy_mu, sig_val=0.025, visualize=False):
        """
        Runs the transformations and deglitching algorithm on mu and returns
        the deglitched mu.
        
        First, the given mu values are transformed to chi, deglitching is
        performed on chi, and then using the same parameters for forward transformation,
        transforms the deglitched chi back to mu.
        
        Paramters:
        e (array, float): The array of energy values. Used for determining the
        transformation of mu to chi.
        glitchy_mu (array, float): The array of measured absorption coefficient values
        corresponding to the energy values in e. May contain glitches.
        sig_val (float, default=0.025): The parameter passed to deglitch(). See Grubbs_test()
        for details.
        visualize (bool, default=false): If true, will make plots depicting the deglitcing 
        process. Visualization is helpful for testing and debuggin, but not necessary for
        running the function.
        
        Returns:
        deglitched mu (array, float): The array of deglitched absorption coefficient values. 
        """
        
        k, chi=self.transform.forward(e, glitchy_mu)
        
        if visualize:
            
            # The guessed glitchy points are the indices of k, not e.
            (deglitched_chi,
            predictions,
            point_glitches,
            step_glitches)=self.deglitch(chi, sig_val, visualize)
            
            # Some visualization
            plt.figure(2)
            plt.plot(k, chi)
            plt.plot(k, deglitched_chi)
            plt.scatter(k[:-self.out_size], predictions, s=8, color='r')
            print("Possible point glitches at: ", k[point_glitches])
            print("Possible step glitches at: ", k[step_glitches])       
            
        else:
            deglitched_chi=self.deglitch(chi, sig_val, visualize)
            
        deglitched_mu=self.transform.reverse(deglitched_chi, glitchy_mu)
            
        return deglitched_mu
        
        
    def run_twostage(self, e, glitchy_mu, sig_val=0.025, visualize=False):
        """
        Runs the transformations and deglitching algorithm on mu and returns
        the deglitched mu.
        
        Large glitches may result in non-ideal background subtraction during the
        transformation from mu to chi and therefore poorly placed deglitched points,
        so a two-stage process is implemented to eliminate the effect of large glithes
        on background subtraction.
       
        First, the given mu values are transformed to chi, then glitches are identified
        in chi and crudley deglitched in mu. The crudely deglitched mu is then transformed
        to chi again. Using the background subtraction from this latest transformation,
        the original mu is transformed to chi again, but without the effect of the large 
        glitches. This latest chi is deglitched as normal, and transformed back to mu.
        
        Paramters:
        e (array, float): The array of energy values. Used for determining the
        transformation of mu to chi.
        glitchy_mu (array, float): The array of measured absorption coefficient values
        corresponding to the energy values in e. May contain glitches.
        sig_val (float, default=0.025): The parameter passed to deglitch(). See Grubbs_test()
        for details.
        visualize (bool, default=false): If true, will make plots depicting the deglitcing 
        process. Visualization is helpful for testing and debuggin, but not necessary for
        running the function.
        
        Returns:
        deglitched mu (array, float): The array of deglitched absorption coefficient values. 
        """    
        
        # Get the first transformation to chi
        k, chi1 = self.transform.forward(e, glitchy_mu)
        
        # Find the glitches as normal
        (deglitched_chi1,
         predictions,
         point_glitches,
         step_glitches)=self.deglitch(chi1, sig_val, return_all=True)
        
        # Do a rough deglitch at the potential problem points
        mu2=np.copy(glitchy_mu)
        for ind in point_glitches:
            ind+=self.transform.e0_idx
            mu2[ind]=(mu2[ind-1]+mu2[ind+1])/2
        for ind in step_glitches:
            ind+=self.transform.e0_idx
            mu2[ind:]-=mu2[ind]-mu2[ind-1]
            
        # Visualization
        if visualize:
            plt.figure(2)
            plt.plot(e, glitchy_mu)
            plt.plot(e, mu2)
        
        # Transform again using the roughly deglitched mu
        k, chi2 = self.transform.forward(e, mu2)
        
        # Transform the original mu using the spline, etc. from the previous
        # transformation
        k, chi3 = self.transform.forward(e, glitchy_mu, freeze_params=True)
        
        # Deglitch as normal
        if visualize:
            (deglitched_chi,
            predictions,
            point_glitches,
            step_glitches)=self.deglitch(chi3, sig_val, visualize)
            
            plt.figure(3)
            plt.plot(k, chi3)
            plt.plot(k, deglitched_chi)
            plt.scatter(k[:-self.out_size], predictions, s=8, color='r')
            print("Possible point glitches at: ", k[point_glitches])
            print("Possible step glitches at: ", k[step_glitches])
            
        else:
            deglitched_chi=self.deglitch(chi3, sig_val, visualize)
        
        # Transform back to mu
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
    #start=58
    #print("Start: %d"%start)
    #es=E[start*NPIX:(start+1)*NPIX]
    #mus=np.stack(MU[start*NPIX:(start+1)*NPIX])
    #good_pix_inds=find_good_pix(mus)
    #mu=np.sum(mus[good_pix_inds], axis=0)
    #e=np.sum(es[good_pix_inds], axis=0)/len(good_pix_inds)
    ind=np.random.randint(0,len(MU))
    mu=MU[ind]
    e=E[ind]
    # Make sure it doesn't load an empty scan.
    while mu[0]!=mu[0] or max(mu)==0:
        ind=np.random.randint(0,len(MU))
        mu=MU[ind]
        e=E[ind]
    
    glitchy=np.copy(mu)
    
    # Add glitches if desired
    add_glitches=True
    chunk_size=16
    if add_glitches:
        glitchy, step_glitch_ind =add_step_glitch_clean_start(glitchy,
                                            min_size=max(glitchy)/15,
                                            max_size=max(glitchy)/10,
                                            skip_num_points=chunk_size,
                                            return_ind=True)
        glitchy, point_glitch_inds=add_point_glitch_clean_start(glitchy,
                                            min_size=max(glitchy)/20,
                                            max_size=max(glitchy)/15,                                             
                                            skip_num_points=chunk_size,
                                            return_inds=True)
    
    
    plt.figure(1)
    plt.plot(e, mu, label='Original mu')
    plt.plot(e,glitchy, label='With added glitches')

    # Here's the syntax for the use of the deglitcher
    Deglitcher=Mu_Deglitcher()    
    t0=time.time()
    deglitched_mu = Deglitcher.run_twostage(e, glitchy, sig_val=0.01, visualize=True)
    t1=time.time()
    print("That took %.3f seconds."%(t1-t0))
    plt.figure(1)
    plt.plot(e, deglitched_mu, label='Deglitched')
    plt.legend()
    
    
