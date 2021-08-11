"""
Defines the classes and functions used for identifying and removing point and 
step glithes in mu(e). The cropping point currently must be chosen by the user
before attempting deglitching. In a future version, a more intelligent way of 
determining the best cropping point may be used.

Further functions are defined to help create an artificial example  of a glitchy
signal.

The main script runs an example of deglitching and prints and plots the results.
"""


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.stats as stats
from scipy.interpolate import LSQUnivariateSpline

from LSTM_on_chi import LSTMModel



class Mu_Transform():
    """
    Performs forward and reverse transformations on mu to make
    it amenable to next point predictions with the LSTM.
    
    __init__() defines the local variables with null values
    
    forward() transforms mu into its form for predictions muT
    
    reverse() transforms muT back to mu using the same parameters 
    used in the forward transformation.
    """
    
    def __init__(self):
        super(Mu_Transform, self).__init__()
        """
        Defines local variables in case they are called before assignment.
        
        spl (array, float): The spline curve fitted to the cropped mu.
        
        offset (float): The constant term used in the rescaling of mu
        
        scale (float): The coefficient used in the rescaling of mu.
        
        crop_ind (int): The index before which values of mu are ignored for
        the transformation.
        """
        
        self.spl=0
        self.offset=0
        self.scale=1
        self.crop_ind=0
        
        
    def forward(self, e, mu, crop_ind=0, freeze_params=False):
        """
        Transforms mu to muT by cropping mu, rescaling it to be between 0 and 1,
        and then subtracting a spline. 
        
        Parameters:
            
            e (array, float): The energy values corresponding to mu
            
            mu (array, float): The absorption coefficient values corresponding
            to e.
            
            crop_ind (int, default=0): The index before which values of mu are 
            ignored for the transformation.
            
            freese_params (bool, default=False): If true, prevents scale, offset,
            and spl from changing from their previously set values during the 
            transformation.
            
        
        Returns:
            
            ecrop (array, float): The cropped array of energy values.
            
            muT (array, float): The transformed array of mu values.
        """
                
        # Update the cropping index in local variables.
        self.crop_ind=crop_ind
        
        # Define new cropped arrays 
        ecrop=e[crop_ind:]
        muT=mu[crop_ind:]
        
        # Do not update the self.* variables if freeze_params is true.
        if not freeze_params:
            self.offset=min(muT) 
            self.scale=max(muT)-min(muT)
            
        muT=(muT-self.offset)/self.scale # Shift and rescale the values to go
        #from zero to one.
        
        if not freeze_params:
            t=ecrop[len(ecrop)//5::len(ecrop)//5] # Make four knot locations
                # evenly spaced in the indeces. Knot density mimics energy 
                # sampling rate.
            spl_func=LSQUnivariateSpline(ecrop, muT, t, k=3) # Fit a spline
                # using the knot locations and cropped indeces
            
            self.spl=spl_func(ecrop) # Update self.spl if freeze_params is false

        # Subtract the spline to center the oscillations of mu to zero
        muT-=self.spl
        
        # Return the cropped energy values and the transformed mu.        
        return ecrop, muT


    def reverse(self, mu, muT):
        """
        Transform muT to mu using the same spline and rescaling coefficients 
        used to transform mu to muT.
        
        
        Parameters:
            
            mu (array, float): The original array of absorption coefficient 
            values.
            
            muT (array, float): The mu transformed by forward().
            
            
        Returns:
            
            mu (array, float): The reverse transformed mu with the same length
            and range of values as the original mu.
        """
        
        # Add the spline and scale and shift it back
        muT=(muT + self.spl)*self.scale+self.offset
        
        # Stitch the part of mu that was cropped out with the reverse
        # transformed part
        mu=np.concatenate([mu[:self.crop_ind], muT])

        return mu
        


class Mu_Deglitcher():    
    """
    Wraps all the functions necessary to perform the deglitching on mu.
    
    Variables:
        
    device ('cpu' or 'cuda'): The device to be used when running the model 
    on the chunks. 'cpu' is the default because the program runs faster for 
    deglitching a single mu.
    
    chunk_size (int): The number of points in each chunk. Default is 16. 
    For the model currently being loaded, chunk_size must be 16.
    
    out_size (int): The number of floats outputted by the model. Default
    is 1. For the current model, out_size must be 1.
    
    model (pytorch module): The LSTM model used for predicting the next
    point in the sequence.
    
    transform (class): The class used to transform mu for deglitching and back 
    again. Default is Mu_transform(). It must contain .forward() and .reverse()
    which return the respective transformations. .forward() must have the 
    keyword boolean argument 'freeze_params' and the keyword argument crop_ind.
    
    
    Functions:
        
    Grubbs_test: Performs Grubb's test on an array
    
    deglitch: Removes sharp points and sudden steps from an array.
    
    run: Performs the transformations and deglitching.
    
    run_twostages: Performs the transformations and deglitching
    with a two-stage process.
    """
    
    def __init__(self):
        """
        Loads the model, initializes variables, and defines the transform class.
        """
        
        # Load the model
        hidden_size=32        
        batch_size=1
        num_layers=2
        bidirectional=False
        drop_prob=0.5
        
        #self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.device='cpu'
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
        self.model.load_state_dict(torch.load("lstm_mu2.pth",
                                              map_location=torch.device(self.device)))
        self.model.init_hidden(batch_size)
        
        # Define the transform class
        self.transform=Mu_Transform()
        
        

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
        
        mean=np.mean(r) # Calculate the mean of the residuals 
        std=np.std(r) # Calculate the standard deviation
        max_ind=np.argmax(abs(r-mean)) # Find the index of the residual furthest
            # from the mean
    
        test_stat=abs(r[max_ind]-mean)/std # Distance of the residual from the
        # mean (of the residuals) in units of standard deviations

        # Some intermediate values
        df=len(r)
        p=1-sig_val/df/2 
        t=stats.t.ppf(p, df-2)
        critical_val=(df-1)*t/np.sqrt((df-2+t*t)*df) 
        # Maximum acceptable distance from the mean in units of standard deviations
    
        # Return the index of the outlier
        if test_stat>critical_val:
            return max_ind
        else:
            return None
        
        

    def deglitch(self,
                 e,
                 glitchy,
                 sig_val=0.025, 
                 return_all=False):
        """
        Takes a glitchy signal, and deglitches it using the model provided. 
        
        A trained LSTM model is used to take sets of consecutive values from
        a signal and predict the next value in the sequence. The difference 
        between the predicted value and the next measured value is compared to
        the most recent set of differences in predictions and measured values. 
        If the latest difference between the predicted value and the measured 
        value is determined to be an outlier by Grubb's test, then that point 
        is labelled as a glitch. The measured value is replaced by the predicted
        value. If the following measured value is determined to be an outlier 
        in the same way, the glitch is labelled as a step glitch and the rest 
        of the signal is shifted by the difference between the predicted value 
        and the measured value to fix it. If the following point is not an 
        outlier, then it was a point glitch and was already replaced by the 
        corresponding predicted value as a fix. 
        
        Instead of Grubb's test, a threshold method can be used to identify 
        outliers. The threshold is the maximum absolute difference a measured
        value can have from the predicted value for it to be labelled as normal 
        data. The threshold is calculated as five times the root mean squared 
        error between the most recent set of predicted values and their measured
        values.
        
        For most sets of measured mu, the energy sampling rate has abrupt changes
        which can trigger a false glitch detection. This algorithm also finds
        these sharp changes and does not allow for glitch detection within
        half the chunk_size number of points after the change.
        
        
        Parameters:
            
        glitchy (array, float): The set of values (the signal) that is to be 
        deglitched.
        
        sig_val (float, default=0.025): The significance value with which 
        Grubb's test is conducted. See Grubbs_test() for details.
        
        return_all (bool, default=False): If false, deglitched is returned.
        If true, deglitched, predictions, point_glitches, and step_glitches are
        returned.
        
        
        Returns:
            
        deglitched (float, array): The deglitched signal.
        
        predictions (float, list): The list of predictions made by the model 
        during deglitching.
        
        point_glitches (int, list): The list of indeces determined to contain 
        point glitches.
        
        step_glitches (int, list): The list of indeces determined to contain 
        step glitches.
        """
        
        
        # The sharp changes in sampling rate can cause false positives in the
        # glitch detection, so these lines find the sharp changes
        # and the deglitching algorithm is hard coded to not detect glitches
        # for a few points after the sharp changes.
        facts=np.exp(np.abs(np.log((e[2:]-e[1:-1])/(e[1:-1]-e[:-2])))) # Factors
            # by which consecutive sampling rates change. The absolute value of
            # the ratios are taken in log space to enforce that the ratio is 
            # greater than 1. 
        bad_inds=np.where(facts>2)[0] # The indeces at which the sampling rate
            # changes by more than a factor of two.
            
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
        predictions=list(deglitched[:self.chunk_size]+\
                         np.random.normal(0, 0.02, self.chunk_size))
                
        
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
            
            # A prediction still needed to be made for the next point, but now
            # move on to the next index if a sharp change in the sampling rate
            # is too recent.
            if ((ind-bad_inds<chunk_size//2) & (ind>=bad_inds)).any():
                continue
            
            
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
                
                chunk_glitch_ind=self.Grubbs_test(r, sig_val/8)
                
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
        
        
    
    
    def run(self, e, glitchy_mu, crop_ind=0, sig_val=0.025, visualize=False):
        """
        Runs the transformations and deglitching algorithm on mu and returns
        the deglitched mu.
        
        First, the given mu values are transformed, deglitching is performed on 
        the transformed mu, and then using the same parameters for forward 
        transformation, the deglitched mu in transformed back into the original
        space.
        
        
        Paramters:
            
        e (array, float): The array of energy values. Used to account for the 
        changes in energy sampling rate.
        
        glitchy_mu (array, float): The array of measured absorption coefficient
        values corresponding to the energy values in e. May contain glitches.
        
        crop_ind (int, default=0): The index before which values of mu are 
        ignored for the transformation and deglitching.
        
        sig_val (float, default=0.025): The parameter passed to deglitch().
        See Grubbs_test() for details.
        
        visualize (bool, default=false): If true, will create plots depicting 
        the deglitcing process. Visualization is helpful for testing and 
        debugging, but not necessary for running the function.
        
        
        Returns:
            
        deglitched_mu (array, float): The array of deglitched absorption 
        coefficient values. 
        """
        
        # Make sure the energy step is always positive.
        diffTest = np.diff(e) <= 0
        if any(diffTest):
            e[np.where(diffTest)] -= 1e-1
        
        
        ecrop, muT=self.transform.forward(e, glitchy_mu, crop_ind)
        
        if visualize:            
            # The guessed glitchy points are the indices of k, not e.
            (deglitched_muT,
            predictions,
            point_glitches,
            step_glitches)=self.deglitch(ecrop, muT, sig_val, visualize)
            
            # Some visualization
            plt.figure(3)
            plt.plot(ecrop, muT, label='Original')
            plt.plot(ecrop, deglitched_muT, label='Deglitched')
            plt.scatter(ecrop[:-self.out_size], predictions, s=8, color='r', label='Predictions')
            plt.legend()
            print("Possible point glitches at: ", ecrop[point_glitches])
            print("Possible step glitches at: ", ecrop[step_glitches])       
            
        else:
            deglitched_muT=self.deglitch(ecrop, muT, sig_val, visualize)
            
        deglitched_mu=self.transform.reverse(glitchy_mu, deglitched_muT)
            
        return deglitched_mu
        
        
    
    def run_twostage(self, e, glitchy_mu, crop_ind=0, sig_val=0.025, visualize=False):
        """
        Runs the transformations and deglitching algorithm on mu and returns
        the deglitched mu.
        
        Large glitches may result in non-ideal spline subtraction during the
        transformation of mu and therefore deglitched points will not be ideal.
        So a two-stage process is implemented to eliminate the effect of large 
        glithes on spline fitting.
       
        First, the given mu values are transformed, then glitches are identified
        and crudley deglitched in the original space. The crudely deglitched mu 
        is then transformed again. Using the spline subtraction from this latest
        transformation, the original mu is transformed again, but without the 
        effects of the large glitches. This latest transformed mu is deglitched 
        as normal, and transformed back.
        
        
        Paramters:
            
        e (array, float): The array of energy values. Used to account for the 
        changes in energy sampling rate.
        
        glitchy_mu (array, float): The array of measured absorption coefficient
        values corresponding to the energy values in e. May contain glitches.
        
        crop_ind (int, default=0): The index before which values of mu are 
        ignored for the transformation and deglitching.
        
        sig_val (float, default=0.025): The parameter passed to deglitch().
        See Grubbs_test() for details.
        
        visualize (bool, default=false): If true, will create plots depicting 
        the deglitcing process. Visualization is helpful for testing and 
        debugging, but not necessary for running the function.
        
        
        Returns:
            
        deglitched_mu (array, float): The array of deglitched absorption 
        coefficient values. 
        """

        # Make sure the energy step is always positive.        
        diffTest = np.diff(e) <= 0
        if any(diffTest):
            e[np.where(diffTest)] -= 1e-1
        
        # Get the first transformation
        ecrop, muT1 = self.transform.forward(e, glitchy_mu, crop_ind)
        
        # Find the glitches as normal
        (deglitched_muT1,
         predictions,
         point_glitches,
         step_glitches)=self.deglitch(ecrop, muT1, sig_val, return_all=True)
        
        # Do a rough deglitch at the potential problem points
        mu2=np.copy(glitchy_mu)
        for ind in point_glitches:
            ind+=crop_ind # The indeces given by the deglitching algorithm
                # are crop_ind less than the indeces that refer to the glitches
                # in mu
            mu2[ind]=(mu2[ind-1]+mu2[ind+1])/2 # Replace point glitches with 
                # average of the neighbouring points
        for ind in step_glitches:
            ind+=crop_ind
            mu2[ind:]-=mu2[ind]-mu2[ind-1] # Shift step glitches by the differnce
                # between the last point and this one.
            
        # Visualization
        if visualize:
            plt.figure(3)
            plt.plot(e, glitchy_mu, label="Original")
            plt.plot(e, mu2, label="Roughly deglitched") 
            plt.legend()
        
        # Transform again using the roughly deglitched mu
        ecrop, muT2 = self.transform.forward(e, mu2, crop_ind)
        
        # Transform the original mu using the spline, etc. from the previous
        # transformation
        ecrop, muT3 = self.transform.forward(e, glitchy_mu, crop_ind, freeze_params=True)
        
        # Deglitch as normal
        if visualize:
            (deglitched_muT,
            predictions,
            point_glitches,
            step_glitches)=self.deglitch(ecrop, muT3, sig_val, visualize)
            
            plt.figure(4)
            plt.plot(ecrop, muT3, label='Transformed mu')
            plt.plot(ecrop, deglitched_muT, label='Deglitched transformed mu')
            plt.scatter(ecrop[:-self.out_size], predictions, s=8, color='r', label='Predictions')
            plt.legend()
            print("Possible point glitches at: ", ecrop[point_glitches])
            print("Possible step glitches at: ", ecrop[step_glitches])
            
        else:
            deglitched_muT=self.deglitch(ecrop, muT3, sig_val, visualize)
        
        # Transform back to original space
        deglitched_mu=self.transform.reverse(glitchy_mu, deglitched_muT)
        
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
                        (max_size-min_size)+min_size)   
    
    if return_inds:
        return y_glitch, randinds
    else:
        return y_glitch
    





if __name__=="__main__":
    
  

    # Load an example mu and e
    # Just an example I have on file
    MU=np.load("pixel_mus.npy", allow_pickle=True)
    E=np.load("pixel_es.npy", allow_pickle=True)
    #NPIX=32
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
    while mu[0]!=mu[0] or max(mu)==0:
        ind=np.random.randint(0,len(MU))
        mu=MU[ind]
        e=E[ind]

    #mu_dir="mu_test"
    #mulist=os.listdir(mu_dir)
    #muname=np.random.choice(mulist)
    #mu=np.genfromtxt(os.path.join(mu_dir, muname))
    #e=np.arange(len(mu))

    clean=np.copy(mu)
    #clean=torch.from_numpy(mu)
    #e=torch.from_numpy(e)
    #m = nn.Upsample(size=int(1*len(clean)), mode='linear')
    #clean=m(clean.view(1,1,-1)).squeeze().numpy()
    #e=m(e.view(1,1,-1).float()).squeeze().numpy()
    #glitchy=clean+np.random.normal(0,(max(clean)-min(clean))/200, len(clean))
    glitchy=clean.copy()
    
    #glitchy=np.copy(mu)
         
    # Add glitches if desired
    add_glitches=False
    chunk_size=16
    mcglitch=np.array([-0.06666667, 
                               -0.33333333, 
                               -1.00000000, 
                               0.33333333, 
                               0.11111111])
    if add_glitches:
        glitchy, step_glitch_ind =add_step_glitch_clean_start(glitchy,
                                            min_size=max(glitchy)/12,
                                            max_size=max(glitchy)/9,
                                            skip_num_points=chunk_size,
                                            return_ind=True)
        glitchy, point_glitch_inds=add_point_glitch_clean_start(glitchy,
                                            min_size=max(glitchy)/12,
                                            max_size=max(glitchy)/9,                                             
                                            skip_num_points=chunk_size,
                                            return_inds=True)
    
#        glitchy, mc_glitch_ind = add_mcglitch(glitchy,
#                                              mcglitch,
#                                              min_size=max(glitchy)/12,
#                                              max_size=max(glitchy)/9,
#                                              skip_num_points=chunk_size,
#                                              return_inds=True)
    
    
    plt.figure(1, figsize=(5,4))
    plt.xlabel("X-ray energy (eV)")
    plt.ylabel("Absorption coefficient (normalized)")
    plt.plot(e, glitchy, label='Measured')
    #plt.plot(e,glitchy, label='With added glitches')
    

#==============================================================================
    # Here's the syntax for the use of the deglitcher
    
    crop_ind=75
    # To define the cropping point with a particular energy
    #e0=9000 # eV
    #crop_ind=sum(e<e0)
    Deglitcher=Mu_Deglitcher()    
    t0=time.time()
    deglitched_mu = Deglitcher.run_twostage(e,
                                            glitchy, 
                                            crop_ind=crop_ind, 
                                            sig_val=0.01, 
                                            visualize=True)
    t1=time.time()
#==============================================================================    
    
    
    plt.figure(1)
    plt.plot(e, deglitched_mu, label='Deglitched')
    plt.plot(e[crop_ind:], Deglitcher.transform.spl*Deglitcher.transform.scale+Deglitcher.transform.offset, label="Spline")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("Deglitching example.png", dpi=400)
    print("That took %.3f seconds."%(t1-t0))
    
