"""
Defines the classes and functions used for identifying and removing point and 
step glithes in mu(e). The cropping point currently must be chosen by the user
before attempting deglitching. In a future version, a more intelligent way of 
determining the best cropping point may be used.

Further functions are defined to help create an artificial example  of a glitchy
signal.

The main script runs an example of deglitching and prints and plots the results.
Several chunks of code in main() are commented out. These provide different
examples of mu and e.
"""


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import scipy.stats as stats
from scipy.interpolate import LSQUnivariateSpline
from scipy.interpolate import interp1d

from LSTM_on_mu import LSTMModel
from discriminate_pixels import *



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
            de=len(ecrop)//5
            t=ecrop[de:-de:de] # Make four knot locations
                # evenly spaced in the indeces. Knot density mimics energy 
                # sampling rate.
                
            #t=np.linspace(min(ecrop)+(max(ecrop)-min(ecrop))/10, max(ecrop)-(max(ecrop)-min(ecrop))/10, 5)
                
            spl_func=LSQUnivariateSpline(ecrop, muT, t, k=3)#, w=np.exp(np.linspace(0,2,len(e_spl))))
            # Fit a spline
                # using the knot locations and cropped indeces
            
            self.spl=spl_func(ecrop) # Update self.spl if freeze_params is false

        # Subtract the spline to center the oscillations of mu to zero
        muT-=self.spl
        
        if not freeze_params:
            self.scale2=max(muT)-min(muT)
            
        #muT/=self.scale2
        
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
        #muT*=self.scale2
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
    
    model_low_sampling (pytorch module): The LSTM model used for predicting the
    next point in a sequence of relatively low sampled data.
    
    model_high_sampling (pytorch module): The LSTM model used for predicting the
    next point in a sequence of relatively high sampled data.
    
    transform (class): The class used to transform mu for deglitching and back 
    again. Default is Mu_transform(). It must contain .forward() and .reverse()
    which return the respective transformations. .forward() must have the 
    keyword boolean argument 'freeze_params' and the keyword argument crop_ind.
    
    points_removed (int, list): A list of indeces of data points that were
    removed if the difference in energy between it and the next point is too
    small.
    
    
    Functions:
        
    Grubbs_test: Performs Grubb's test on an array
    
    deglitch: Removes sharp points and sudden steps from an array.
    
    deglitch_desampling: Deglitches the spectrum in two parts according to the
    different sampling rates present in the spectrum.
    
    run: Performs the transformations and deglitching.
    
    run_twostages: Performs the transformations and deglitching with a two-stage 
    process.
    """
    
    def __init__(self):
        """
        Loads the models, initializes variables, and defines the transform class.
        """
        
        # Define mutual parameters both models
        hidden_size=32        
        batch_size=1
        num_layers=2
        bidirectional=False
        drop_prob=0.5
        
        #self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.device='cpu'
        self.chunk_size=16
        self.out_size=1
        
        # Load and initialize the model used for making predictions on low 
        # sampling data
        self.model_low_sampling=LSTMModel(self.chunk_size, 
                             hidden_size, 
                             self.out_size, 
                             batch_size,
                             num_layers, 
                             drop_prob,
                             bidirectional,
                             self.device).to(self.device)
        self.model_low_sampling.load_state_dict(torch.load("lstm_mu_1x_sampling.pth",
                                              map_location=torch.device(self.device)))
        self.model_low_sampling.init_hidden(batch_size)
        
        # Load and initialize the model used for high sampling data
        self.model_high_sampling=LSTMModel(self.chunk_size, 
                             hidden_size, 
                             self.out_size, 
                             batch_size,
                             num_layers, 
                             drop_prob,
                             bidirectional,
                             self.device).to(self.device)
        self.model_high_sampling.load_state_dict(torch.load("lstm_mu_1-5x_sampling.pth",
                                              map_location=torch.device(self.device)))
        self.model_high_sampling.init_hidden(batch_size)
        
        # Define the transform class
        self.transform=Mu_Transform()
        
        # Initialize the list of removed indices
        self.points_removed=[]
        
        # Initialize list of deglitched indices
        self.point_glitch_inds=[]
        self.step_glitch_inds=[]
        
        

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
                 glitchy,
                 model,
                 sig_val=0.025, 
                 return_all=False,
                 start_ind=0,
                 end_ind=None):
        """
        Takes a glitchy signal and deglitches a given range of it using the 
        model provided. 
        
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
        
        
        Parameters:
            
        glitchy (array, float): The set of values (the signal) that is to be 
        deglitched.
        
        model (function): Takes in a chunk of an array and outputs a single 
        value.
        
        sig_val (float, default=0.025): The significance value with which 
        Grubb's test is conducted. See Grubbs_test() for details.
        
        return_all (bool, default=False): If false, deglitched is returned.
        If true, deglitched, predictions, point_glitches, and step_glitches are
        returned.
        
        start_ind (int, default=0): The index at which to begin deglitching. 
        Predictions must still be made for earlier points, but the function
        will not make glitch fix changes.
        
        end_ind (int, default=None): The index at which to stop deglitching.
        The point before end_ind is actually the last point that can be
        deglitched. Predictions after end_ind are filled out with the values
        of the rest of the deglitched array.
        
        
        Returns:
            
        deglitched (float, array): The deglitched signal. Always the same length
        as the original signal regardless of the range given by start_ind and
        end_ind.
        
        predictions (float, list): The list of predictions made by the model 
        during deglitching. Also the same length as the original signal.
        
        point_glitches (int, list): The list of indeces determined to contain 
        point glitches.
        
        step_glitches (int, list): The list of indeces determined to contain 
        step glitches.
        """
        
        deglitched=glitchy.copy()
        # If no end_ind is provided, then it is None and we can set it to the 
        # last available index of the array.
        if not end_ind:
            end_ind=len(deglitched)-1
            
        # Reset the hidden and cell states of the LSTM cell
        model.init_hidden(1)
        
        # Initialize lists to track the locations of the glitches
        point_glitches=[]
        step_glitches=[]
        predictions=[]
        

        # FOR GRUBB'S TEST
        # Grubb's test requires a short history of residuals. We obviously can't
        # make predictions before the chunk_size+1th point, so the predictions 
        # are initialized by adding noise to the actual signal
        predictions=list(deglitched[:self.chunk_size]+\
                         np.random.normal(0, 0.01, self.chunk_size))
                
        
        # ind is the index of the value of the signal to be predicted
        # Cycle over every point after ind=chunk_size to look for and fix glitches
        # Don't make a prediction for the last point, becasue a possible step
        # glitch requires there to be another point for comparison
        for ind in range(self.chunk_size, end_ind+1-self.out_size):
            
            # ychunk: the most recent chunk_size values from the signal that are
            # used to make a prediction
            ychunk=torch.Tensor(deglitched[ind-self.chunk_size:ind]).float().detach()
                       
            # val: the value of the signal immediately after the last value in chunk
            val=deglitched[ind]
            
            # THRESHOLD METHOD
            # The threshold is 5 times the root mean squared error
            # of the predictions relative to the values for the last chunk
            #if ind<2*chunk_size:
            #    threshold=0.4
            #else:
            #    threshold=max(0.1,5*np.sqrt(np.mean((predictions[-chunk_size:]-chunk.squeeze().numpy())**2)))
            
            # prediction: the value after the signal chunk predicted by the model
            prediction=model(ychunk.view(1,1,-1).to(self.device))\
                                                        .squeeze().item()
            predictions.append(prediction)
            
            # GRUBB'S TEST METHOD
            # r: the residuals from between last chunk_size+1 predictions and the 
            # given signal
            r=np.array(predictions[-(self.chunk_size+1):])-\
                                      np.append(ychunk.squeeze().numpy(), val)
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
            # from the chunk_size values before that, then it is a glitch.
            # Only attempt to identify and fix glitches if ind is at least
            # at start_ind.
            if chunk_glitch_ind==self.chunk_size and ind>=start_ind:
                
                # Try replacing the problem point with the predicted value
                next_ind=ind+1
                deglitched[ind]=prediction
    
                # Make a prediction for the next point 
                ychunk2=torch.Tensor(\
                    deglitched[next_ind-self.chunk_size:next_ind]\
                    ).float().detach()
                
                prediction2=model(\
                                  ychunk.view(1,1,-1).to(self.device)\
                                  ).squeeze().item()
                
                val2=deglitched[next_ind]
                
                r=np.append(predictions[-self.chunk_size:], prediction2)\
                                -np.append(ychunk2.squeeze().numpy(), val2)
                
                # False step glitches are very disruptive, so the sig_val
                # for step glitch identification is lowered. 
                chunk_glitch_ind=self.Grubbs_test(r, sig_val/10)
                
                # THRESHOLD METHOD
                # If the next point is still far from the prediction
                # It is probably a step glitch
                #if abs(prediction2-val2)>threshold:
                
                # GRUBB'S TEST METHOD
                if chunk_glitch_ind==self.chunk_size:
                    # If the next point was identified as an outlier as well,
                    # then the first point was a step glitch.
                
                    # Subtract the inital step from the rest of the spectrum
                    deglitched[ind]=val
                    step=prediction-val
                    deglitched[ind:]+=step                    
                    
                    # Record the index of the glitch
                    step_glitches.append(ind)
                
                # Then if the next point is close to the predicted value
                # then it was probably a point glitch and has been fixed
                else:
                    # Record the index of the glitch
                    point_glitches.append(ind)
                    
        # Make one last prediction so that the prediction list is the same 
        # length as the start_ind to end-ind interval
        ychunk=torch.Tensor(deglitched[-self.chunk_size-1:-1]).float().detach()
        prediction=model(ychunk.view(1,1,-1).to(self.device)).squeeze().item()
        predictions.append(prediction)
        
        # Fill in the rest of the predictions list with the deglitched values.
        predictions.extend(deglitched[end_ind+1:])
                
                    
        if return_all:
            return deglitched, predictions, point_glitches, step_glitches
        else:
            return deglitched
        
        
        
    def deglitch_desampling(self,
                            e,
                            glitchy,
                            sig_val=0.025,
                            return_all=False):
        """
        Deglitches an array in two parts to better account for a sharp decrease
        in energy sampling rates.
        
        First, the point at which the sampling rate decreases by more than a 
        factor of two is found. Glitches are found and fixed in the high 
        sampling part of the signal with a model trained to find glitches in
        data with that sampling, then the deglitched signal is desampled by a
        factor of 2 in the high sampling region and the desampled signal is 
        deglitched with a model trained for that sampling rate. The two 
        deglitched regions are stitched together and returned as the deglitched
        array. 
        
        If no sharp change in sampling rate is detected, then the signal is
        deglitched normally with the deglitch function and the model trained
        for low sampled data.
        
        Note, the desampled region is only used to inform the model of the
        behaviour of that region and no glitches are found or fixed there. 
        
        Also, this function has nearly identical functionality as deglitch(). 
        Both return the deglitched signal and optionally the predictions made
        and the indeces of the glitches found. The differences are that 
        deglitch_desampling() requires the energy array as an argument and does
        not take in a range of indeces for deglitching.
        
        
        Parameters:
            
            e (array, float): The array of energy values.
            
            glitchy (array, float): The array to be deglitched. Must be the same
            size as e.
            
            sig_val (float, default=0.025): The significance value for outlier
            identification.
            
            return_all(bool, default=False): If true, returns the list of 
            predictions and indeces of point and step glitches as well.
            
            
            
        Returns:
            
            deglitched (float, array): The deglitched signal. 
        
            predictions (float, list): The list of predictions made by the model 
            during deglitching. Always the same length as the original signal.
        
            point_glitches (int, list): The list of indeces determined to contain 
            point glitches.
        
            step_glitches (int, list): The list of indeces determined to contain 
            step glitches.
        """
        

        facts=(e[2:]-e[1:-1])/(e[1:-1]-e[:-2]) # Factors
            # by which consecutive sampling rates change. 
        bad_inds=np.where(facts<0.5)[0] # The indeces before which the sampling rate
            # decreases by more than a factor of two. bad_inds+1 is the index of 
            # the point in between the two sampling rates.
        
        # Only deglitch in two parts if there is a sharp change in sampling rate.
        if len(bad_inds)>0:
             
            # Deglitch the spectrum with the high sampling rate-trained model.
            # Only the first part of the spectrum needs to be searched for 
            # glitches at this point, but changes to the spectrum by step glitch
            # fixes need to be carried forward into the next steps.
            (deglitched1,
             predictions1,
             pg_inds1,
             sg_inds1)=self.deglitch(glitchy, 
                             self.model_high_sampling, 
                             sig_val, 
                             True,
                             0,
                             bad_inds[0]+1)
            
            # Modify the deglitched mu by desampling the beginning to better 
            # match the sampling rate of the rest of the spectrum
            glitchy_dsmpl=np.concatenate((deglitched1[bad_inds[0]-1::-2][::-1],\
                                          deglitched1[bad_inds[0]+1:]))
            
            # Deglitch the entire spectrum with more uniform sampling
            (deglitched_dsmpl, 
             predictions_dsmpl,
             pg_inds2,
             sg_inds2) = self.deglitch(glitchy_dsmpl,
                             self.model_low_sampling, 
                             sig_val, 
                             True,
                             len(glitchy_dsmpl)-len(glitchy)+bad_inds[0]+1)
            
            # Stitch the two deglitched spectra together at the point where 
            # the sampling rate changes.
            deglitched=np.concatenate((deglitched1[:bad_inds[0]+1], \
                            deglitched_dsmpl[-(len(glitchy)-bad_inds[0]-1):]))
        
            # Stitch the lists of predictions and indeces of glitches together 
            # similarly
            predictions=np.concatenate((predictions1[:bad_inds[0]+1], \
                            predictions_dsmpl[-(len(glitchy)-bad_inds[0]-1):]))
            
            pg_inds=np.concatenate([np.array(pg_inds1),\
            np.array(pg_inds2)+len(glitchy)-len(glitchy_dsmpl)]).astype('int')
    
            sg_inds=np.concatenate([np.array(sg_inds1), \
            np.array(sg_inds2)+len(glitchy)-len(glitchy_dsmpl)]).astype('int')
           
            
        # If there is no sharp change in sampling rate, deglitch normally.
        else:
            (deglitched, 
             predictions, 
             pg_inds, 
             sg_inds) = self.deglitch(glitchy, 
                                        self.model_low_sampling, 
                                        sig_val, 
                                        True)
        
        if return_all:
            return deglitched, predictions, pg_inds, sg_inds
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
            
        e_deglitched (array, float): The array of energy values. Identical to
        the one provided as an argument, but with the points removed with 
        very small energy differences.
            
        deglitched_mu (array, float): The array of deglitched absorption 
        coefficient values. 
        """
        
        # Remove and record points with almost zero energy difference       
        diffTest = np.diff(e) <= 0.1
        if any(diffTest):
            self.points_removed=np.where(diffTest)[0]
            e_deglitched=np.delete(e, self.points_removed)
            glitchy_mu=np.delete(glitchy_mu, self.points_removed)
        else: 
            e_deglitched=e.copy()        
        
        # Crop and transform the energy and mu arrays.
        ecrop, muT=self.transform.forward(e_deglitched, glitchy_mu, crop_ind)
        
        # The guessed glitchy points are the indices of ecrop, not e.
        (deglitched_muT,
         predictions,
         self.point_glitch_inds,
         self.step_glitch_inds)=self.deglitch_desampling(ecrop, muT, sig_val, visualize)
        
        # Save the indeces of the deglitched points in e_deglitched in local 
        # variables
        self.point_glitch_inds = (np.array(point_glitch_inds) + crop_ind).astype('int')
        self.step_glitch_inds = (np.array(step_glitch_inds) + crop_ind).astype('int')
        
        if visualize:            
            
            # Some visualization
            plt.figure(3)
            plt.plot(ecrop, muT, label='Original')
            plt.plot(ecrop, deglitched_muT, label='Deglitched')
            plt.scatter(ecrop, predictions, s=8, color='r', label='Predictions')
            plt.legend()
            print("Possible point glitches at: ", e_deglitched[self.point_glitch_inds])
            print("Possible step glitches at: ", e_deglitched[self.step_glitch_inds])       
                       
        # Transform the signal back. 
        deglitched_mu=self.transform.reverse(glitchy_mu, deglitched_muT)
            
        return e_deglitched, deglitched_mu
        
        
    
    def run_twostage(self, e, glitchy_mu, crop_ind=0, sig_val=0.025, visualize=False):
        """
        Runs the transformations and deglitching algorithm on mu and returns
        the deglitched mu.
        
        Large glitches may result in non-ideal spline subtraction during the
        transformation of mu and therefore deglitched points will not be ideal.
        A two-stage process is implemented to eliminate the effect of large 
        glithes on spline fitting.
       
        First, the given mu values are transformed, then glitches are identified
        and crudley deglitched in the original space. The crudely deglitched mu 
        is then transformed again. Using the spline subtraction from this latest
        transformation, the original mu is transformed again without the 
        effects of the large glitches. This latest transformed mu is deglitched 
        as normal, transformed back, and returned.
        
        
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

        # Remove points with almost zero energy difference       
        diffTest = np.diff(e) <= 0.1
        if any(diffTest):
            self.points_removed=np.where(diffTest)[0]
            e_deglitched=np.delete(e, self.points_removed)
            glitchy_mu=np.delete(glitchy_mu, self.points_removed)
        else: 
            e_deglitched=e.copy()
        
        # Get the first transformation
        ecrop, muT1 = self.transform.forward(e_deglitched, glitchy_mu, crop_ind)
        
        # Find the glitches as normal
        (deglitched_muT1,
         predictions,
         point_glitches,
         step_glitches)=self.deglitch_desampling(ecrop, muT1, sig_val, return_all=True)
        
        # Do a rough deglitch at the potential problem points
        mu2=np.copy(glitchy_mu)
        for ind in point_glitches:
            ind+=crop_ind # The indeces given by the deglitching algorithm
                # are crop_ind less than the indeces that refer to the glitches
                # in mu
            mu2[ind]=(mu2[ind-1]+mu2[ind+1])/2 # Replace point glitches with 
                # average of the neighbouring points
                # There should never be an index error here because the deglitch
                # algorithm can't predict a glitch at the very end or very
                # beginning
        for ind in step_glitches:
            ind+=crop_ind
            mu2[ind:]-=mu2[ind]-mu2[ind-1] # Shift step glitches by the differnce
                # between the last point and this one. This results in two 
                # consecutive points with the same value where there used to be
                # a step glitch.
            
        # Visualization
        if visualize:
            plt.figure(3)
            plt.plot(e_deglitched, glitchy_mu, label="Original")
            plt.plot(e_deglitched, mu2, label="Roughly deglitched") 
            plt.legend()
        
        # Transform again using the roughly deglitched mu to get a better spline.
        ecrop, muT2 = self.transform.forward(e_deglitched, mu2, crop_ind)
        
        # Transform the original mu using the spline, etc. from the previous
        # transformation
        ecrop, muT3 = self.transform.forward(e_deglitched,
                                             glitchy_mu, 
                                             crop_ind, 
                                             freeze_params=True)
        
        # Deglitch as normal
        (deglitched_muT,
         predictions,
         point_glitch_inds,
         step_glitch_inds)=self.deglitch_desampling(ecrop, muT3, sig_val, True)
        
        # Save the indeces of the deglitched points in e_deglitched in local 
        # variables
        self.point_glitch_inds = (np.array(point_glitch_inds) + crop_ind).astype('int')
        self.step_glitch_inds = (np.array(step_glitch_inds) + crop_ind).astype('int')
               
        if visualize:
                    
            plt.figure(4)
            plt.plot(ecrop, muT3, label='Transformed mu')
            plt.plot(ecrop, deglitched_muT, label='Deglitched transformed mu')
            plt.scatter(ecrop, predictions, s=8, color='r', label='Predictions')
            #plt.scatter(ecrop[1:], deglitched_muT[:-1], s=8, color='k', label='Last points')
            plt.legend()
            print("Possible point glitches at: ", e_deglitched[self.point_glitch_inds])
            print("Possible step glitches at: ", e_deglitched[self.step_glitch_inds])
        
        
        # Transform back to original space
        deglitched_mu=self.transform.reverse(glitchy_mu, deglitched_muT)
        
        return e_deglitched, deglitched_mu
    
    
    
    def run_w_sigval_range(self, 
                           e,
                           glitchy,
                           crop_ind):
        """
        Deglitches through a range of significance values, and pick the list of step
        glitch guesses and point glitch guesses that appeared the most times in
        a row. Returns the "optimally" deglitched spectrum and the energy list 
        with problem points removed.
        """
        # Establish a best_deglitched
        best_deglitched=glitchy.copy()
    
        # Keep track of the number of times the current list has appeared in a row
        count=0
        # Keep track of the previous set of guesses
        point_prev=[]
        step_prev=[]
        # The most times so far that an exact list has appeared in a row
        max_consec=0
        
        # Scan through a list of values
        for sig_val in np.logspace(-3.0, -1.5, 10):    
            
            # Perform the deglitching algorithm. A lot of the predicitions are 
            # probably redundant, but depending on how it goes through and performs
            # the deglitching, it will change further predictions
            e_deglitched, deglitched = self.run_twostage(e, 
                                                         glitchy,
                                                         crop_ind,
                                                         sig_val)
            
            point_glitch_guesses=self.point_glitch_inds
            step_glitch_guesses=self.step_glitch_inds
            
            # If this list of guesses is the same as the last one:
            if np.array_equal(point_glitch_guesses, point_prev) and \
                np.array_equal(step_glitch_guesses, step_prev):
                # Bump up the number of times we've seen it consecutively
                count+=1
                
                # If it is also the most we've seen a list consecutively (or tied),
                # record the count, the latest significance value for it, the glitch
                # guesses, and the deglitched spectrum
                if count>=max_consec:
                    max_consec=count
                    best_sigval=sig_val
                    best_points=point_glitch_guesses
                    best_steps=step_glitch_guesses
                    best_deglitched=deglitched
            # If not
            else:
                # reset the counter
                count=0
            
            # Update the list of previous guesses
            point_prev=point_glitch_guesses
            step_prev=step_glitch_guesses
            
            # if best_deglitched was never updated, it won't have the same size
            # as e_deglitched if e contained overlapping energy points.
            if len(best_deglitched)!=len(e_deglitched):
                best_deglitched=np.delete(best_deglitched, self.points_removed)
            
        return e_deglitched, best_deglitched
        


# Functions for adding glitches to simulate glitchy data. Only useful for 
        # demonstrative examples in this script.
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
    
    # This first bunch of code is just loading/creating an example on which to 
    # deglitch

    # Load an example mu and e
    # Just an example I have on file
    MU=np.load("pixel_mus_markers.npy", allow_pickle=True)
    E=np.load("pixel_es_markers.npy", allow_pickle=True)
    
    # Pick one from the list.
    # Make sure it isn't 0 or nan
    ind=np.random.randint(0,len(MU))
    #ind=192
    mu=MU[ind]
    e=E[ind]
    while mu[0]!=mu[0] or max(mu)==0:
        ind=np.random.randint(0,len(MU))
        mu=MU[ind]
        e=E[ind]
    
    # Pick a set of 32 pixel measurements from a fluorescence scan, do the 
    # discrimination and sum them up.
    #NPIX=32
    #start=np.random.choice(range(22))
    #start=8
    #print("Start: %d"%start)
    #es=E[start*NPIX:(start+1)*NPIX]
    #mus=np.stack(MU[start*NPIX:(start+1)*NPIX])
    #good_pix_inds=find_good_pix(mus)
    #mu=np.sum(mus[good_pix_inds], axis=0)
    #e=es[good_pix_inds[0]]
    
    # An example with a step glitch at the point where sampling rates change.
    #dat=np.genfromtxt("C:\\Users\\Me!\\Documents\\CLS 2021\\Machine learning\
    #\\With FEFF data\\Detector channels data\
    #\\Ag K  EXAFS at M1pitch 012 bend 77 blade 1p5_1.txt")
    #dat=np.transpose(dat)
    #e=dat[0]
    #mu=dat[1]/dat[3]
    
    # This code pulls an example mu from the FEFF data I'm using
    #mu_dir="mu_test"
    #mulist=os.listdir(mu_dir)
    #muname=np.random.choice(mulist)
    #mu=np.genfromtxt(os.path.join(mu_dir, muname))
    #e=np.arange(len(mu))

    clean=np.copy(mu)
    # Option to upsample the example for testing
    #clean=torch.from_numpy(mu)
    #e=torch.from_numpy(e)
    #m = nn.Upsample(size=int(1*len(clean)), mode='linear')
    #clean=m(clean.view(1,1,-1)).squeeze().numpy()
    #e=m(e.view(1,1,-1).float()).squeeze().numpy()
    
    # Either add noise or just copy directly.
    #glitchy=clean+np.random.normal(0,0.5, len(clean))
    glitchy=clean.copy()
         
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
    
        # Option to add a mock monochromator glitch.
#        glitchy, mc_glitch_ind = add_mcglitch(glitchy,
#                                              mcglitch,
#                                              min_size=max(glitchy)/12,
#                                              max_size=max(glitchy)/9,
#                                              skip_num_points=chunk_size,
#                                              return_inds=True)
    
    
    # Plot the original mu being used
    plt.figure(1, figsize=(5,4))
    plt.xlabel("X-ray energy (eV)")
    plt.ylabel("Absorption coefficient (normalized)")
    plt.plot(e, glitchy, label='Measured', marker='.')
    #plt.plot(e,glitchy, label='With added glitches')
    

#==============================================================================
    # Here's the syntax for the use of the deglitcher
    # With e and glitchy defined:
    crop_ind=105
    # To define the cropping point with a particular energy:
    #e0=9000 # eV
    #crop_ind=sum(e<e0)
    Deglitcher=Mu_Deglitcher()    
    t0=time.time() # Optional to keep track of how long deglitching takes
    e_deglitched, deglitched_mu = Deglitcher.run_twostage(e,
                                            glitchy, 
                                            crop_ind, 
                                            sig_val=0.0005, 
                                            visualize=True)
    t1=time.time()
#==============================================================================    
    
    # Print the indeces of the removed points
    print("Indeces of points removed: ", Deglitcher.points_removed)

     # Plot the deglitched mu and the spline that was used for the transformation
    plt.figure(1)
    plt.plot(e_deglitched, deglitched_mu, label='Deglitched', marker='.')
    plt.plot(e_deglitched[crop_ind:], \
             Deglitcher.transform.spl*Deglitcher.transform.scale+\
             Deglitcher.transform.offset, label="Spline")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("Deglitching example.png", dpi=400)
    print("That took %.3f seconds."%(t1-t0))
    
