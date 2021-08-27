# EXAFS-deglitching

# 1. deglitch_mu.py and lstm_mu_*x_sampling.pth

Defines the classes used to deglitch mu and provides an example. lstm_mu_*x_sampling.pth refers to a set of trained weights for an LSTM model where the * represents a number with a hyphen in place of a decimal point.

There are three essential algorithms that work to remove the glitches in the data. At the lowest level is the deglitching algorithm that uses the trained model. This is performed by the function `deglitch()`. A trained Long Short-Term Memory (LSTM) neural network is used to make predictions about the value of successive points in the sequence using a chunk of previous points. If the next measured value is statistically significantly far from the value predicted for it, then that point is labelled as a glitch. The glitchy point is replaced with the predicted value to remove the glitch. If the following point is also determined to be a glitch, then it is determined to be the second point in a step glitch and the rest of the signal is shifted by the difference between the first predicted point and the corresponding measured value. The model must make predictions on points consecutively because the LSTM works by storing information about the long-term trends of a sequence in order to inform its predictions.

Statistically significant outliers are determined using Grubb's test on the latest chunk of differences between predictions and their corresponding measured values. For more information on Grubb's test, see https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm or https://en.wikipedia.org/wiki/Grubbs%27s_test. 

Most given measurements of mu have sharp changes in the energy sampling rate. The model is trained on a certain sampling rate and therefore the given data should loosely match what it was trained on. The next algorithm performs deglitching in two parts, done by the function `deglitch_desampling`. First, it uses a model trained on high energy sampling to deglitch the high energy sampling (first) region of the spectrum. Second, it desamples the high sampled region and performs deglitching on the latter part of the spectrum using a model trained for low sampling. Only the originally sparse section is deglitched at this point. The desampling of the earlier section is to initialize the LSTM. The two deglitched sections are then stitched together.

The trained models in this repository have been trained on many different sampling rates. The models all use the same parameters but have different trained weights, which are loaded using the files `lstm_mu_*x_sampling.pth` where * is the factor by which the training data was upsampled. A hyphen is used as a decimal point.

The LSTM has a hard time making predictions on mu as it is, so mu is transformed before deglitching by rescaling it to be between zero and one and then subtracting a fitted spline. This has the effect of centering the oscillations around zero, much like chi, and stabilizes the predictions made by the LSTM. The signal is deglitched in this form and then transformed back with the original transformation parameters. This is the third algorithm, done by the `*.run()` method. However, larger glitches, especially step glitches and especially when they are early in the signal, can greatly affect the fitting of the spline and the deglitched oscillations will not be centered around zero and predictions can be systematically askew. Large glitches will still be identified, but fixing them with the predictions will not be effective. `*.run_twostage()` is another algorithm that accounts for this. First, a spline is fitted as normal and large glitches are identified. Instead of fixing them with the possibly askew predictions, they are fixed crudely by replacing point glitches with average of the two neighbouring points, and shifting step glitches such that the glitchy point is the same as the point before it. A spline fit to the crudely deglitched signal is used to transform the original signal again without strong effects from large glitches. Finally, the signal is deglitched normally with the deglitching algorithm.

### Usage

Define an object with the `Mu_Deglitcher()` class. Use one of `.run()` or `.run_twostage()` with their corresponding arguments and parameters to return the deglitched signal. Inline comments are detailed. Follow the example given.

### Dependencies

Imports LSTMModel from LSTM_on_mu. This is the class that defines the model used for making predictions. When loading the model, deglitch_mu.py uses the file lstm_mu.pth to load all the trained weights and biases. The hyperparamters for the model are defined in the Mu_Deglitcher class and can not be changed for the model to load. 

For an example where the clustering discrimination is conducted to provide mu and e, the script imports the functions from `discriminate_pixels.py`.

### Notes

- It is recommended to use `.run_twostage()` because it more robustly detects and fixes glitches, particularly when large step glitches are present.

- Both `*.run()` and `*.run_twostage()` will check for energy points that are nearly on top of each other and remove one if they exist. This ensures that such points won't interfere during desampling stitching. The indices of any removed points are stored in the `Mu_deglitcher` attribute `points_removed`. 

- `sig_val` is the main tuning parameter. It is essentially a measure of the sensitivity of the algorithm. A higher value of `sig_val` will result in glitch detections coming more easily, and a lower value of `sig_val` will only detect glitches that are more pronounced. With clean data, a good next-point predictor, and glitches that show true and sudden statistically significant deviations from the trend, there should exist a range of values of `sig_val` for which only the glitches are identified. The script "deglitch_sigval_range.py" was an early attempt to run the deglitching algorithm for a range of values for `sig_val` to avoid having to manually find the optimal value, described more below. I have found `sig_val=0.005` is a conservative value to begin with, and it can be increased for more aggressive glitch detection. Higher than `sig_val=0.025` and the algorithm will likely report false positives.
  + Falsely identified step glitches are very disruptive relative to falsely identified point glitches, so the default in the program is to have the significance value for step glitch identification be ten times lower than for point glitches. This factor is tunable, but there is currently no parameter for it in the function. The code itself would need to be changed. I have found that having the program be more careful with step glitches results in very few false negatives and far fewer false positives. Only very rarely will the program mark the edge of a step glitch as a point glitch and not identify the step glitch.

- Using an external GPU to deglitch a single spectrum is not faster than a CPU (atleast the one I have been using). A GPU does improve the speed of training. 

- The algorithms are designed to remove point and step glitches. Removing point glitches is usually very successful, and false positives are typically inconsequential. False positive step glitches can corrupt the signal more, so it is worth double-checking the necessity when a step glitch is reported. Monochromator glitches involve more than one deviant point and the model is not designed to identify or remove them. However, some initial testing has shown that a monochromator glitch can end up being treated as a combination of point and step glitches and may still be removed entirely. A good indication of a monochromator glitch in the signal is several step glitches being reported in consecutive points. There are also other shapes of glitches that may be present. If they begin with sharp enough changes to trigger the glitch detection algorithm, it is common for the glitch to be smoothed out very well with a series of point and step glitch fixes, shown in the example below, where all changes to the original signal were made by step glitch fixes.
However, a glitch beginning with slow changes will not trigger the glitch detection algorithm and can temporarily disrupt the predictions, as seen in the smaller two-point glitch near 9200. If the glitch ends with a sharp change, it could end up shifting the signal up/down to the glitch, which is not desirable. Again, even with different shaped glitches, the algorithm will tend to smooth them out with several step and point glitch fixes, but it could do the opposite. So it is always a good idea to verify the success of the algorithm if several glitches are reported in a row.

![w](multi-point_glitches_example.png)

- The indices of the detected glitches are not returned and are only printed out if `visualize=True`. It may be desired to have the detected glitch indices reported every time with the deglitched signal, or to have them stored as a variable in the deglitcher object. Let me know.

- Cropping off some initial part of the measured mu is not strictly necessary. The deglitching algorithm runs mostly fine with `crop_ind=0`. However, the sharp increases in energy sampling, the transition to the edge jump, and the pre-edge peaks can all cause false positive identifications of step or point glitches. For best results, crop off the measured points until just after the pre-edge peak. This may be earlier than the E_0 used to transform mu to chi, but it may end up being convenient to just use E_0 as the cropping point. In a future version, the choice of cropping point may be automated.

- The current default choices of trained models is 1.5x for the high sampling sections and 1x for the sparse sections. I find that the algorithm works well in this configuration for many different datasets, but in areas with strong oscillations, the next point predictions can drift from the real data in different ways if the data being used for predictions is too different from the training data. There are two major indicators of using a model trained on improper data. If the measured data contains a higher sampling rate than what the model was trained on, then the model will tend to make predictions that are closer to zero than the real data. My guess is that this is because the model expects each oscillation to take a fewer number of points, so it guesses that the signal should be returning to zero sooner than it is. If the measured data contains lower sampling rates than what the model was trained on, then the predictions will tend to lag behind the real data. My guess is that the model is accustomed to guessing smaller changes between points. The result is that poor predictions can cause less consistent glitch detection and less effective glitch fixing. Currently, the two trained models to use for glitch detection must be chosen by trial and error, but the defaults should provide a good start. 

- The hyperparameters for training and the current set of default glitch detection parameters have all been (possibly over-) tuned to work best for the limited set of real world testing data I had access to. This may have resulted in overfitting and results may be worse for different datasets. More testing will be done on more data as soon as possible. 

- deglitch_mu.py currently uses Grubb's test to determine if the difference between the next point and its predicted value is an outlier compared to the differences between the chunk of previous points and their predicted values. However, the file still contains commented-out code that uses a threshold method instead of Grubb's test. The threshold is the minimum absolute difference between a value and its predicted point for the point to be identified as an outlier. The value of the threshold is continuously updated to be five times the root mean squared error of the last chunk of predictions compared to their corresponding measured values. I remember this method working fine, but I haven't used it in a while, and some small modifications would be needed to be made to accommodate the change. For example, a tuning parameter would need to be passed through the functions to vary the sensitivity of the outlier detection instead of `sig_val`. The tuning paramter would probably be the multiplicative factor used in the threshold in front of the root mean squared error of the previous chunk.


### Future work

- As mentioned above, it would be convenient for the program to automatically choose an ideal cropping point. My best guess is that it would just be E_0, or perhaps chunk_size number of points before E_0 so that the entire spectrum after E_0 can be deglitched.

- An automatic choice of `sig_val` would be nice as well. The best `sig_val` can be very different for different measurements. I find that the noise level is one of the biggest influencers on the size of `sig_val`. The higher the noise, the more chances of false positives and therefore the lower `sig_val` should be. While working on the pixel discrimination program, I found that the standard deviation of consecutive differences is a pretty good indicator of the noise level. Specifically, when multiplied by sqrt(2), it gets remarkably close to the real standard deviation of the gaussian noise. Perhaps this figure could be used to inform the significance value.

- The last thing to automate is the choice of which trained models to use. Perhaps it can be assumed that a full scan is taken, and the number of points in each sampling section can be used to decide which of the trained models to use.

- It may be valuable to try training the model on real measured data. This would remove the advantage of training the model against the noiseless spectra, and it wouldn't be possible to confirm that the training data does not contain glitches, but it could give the model a better understanding of the shapes and sampling rates it should expect.

- I was never able to get consistently good predictions from the model without centering the oscillations around zero. If such a transformation could be made unnecessary, or if the spline subtraction could be known ahead of time, then real time glitch detection could be possible with the LSTM. It would probably also be more robust as I have found that the spline subtraction and the choice of cropping index greatly affect the shape of the transformed signal.

- LSTMs have the capability to use future context to inform their predictions about a current point. This is the "bidirectional" feature. To me, this sounds incredibly promising as a way to make much better predictions. However, a necessary part of the deglitching algorithm is that it is known that all the data it uses to make a prediction is free of glitches. I think that there could be problems in the predictions if there are glitches in the future context it is trying to use. Perhaps the LSTM could be trained with glitches in the data to try to learn to ignore them, but I don't know how it could handle step glitches. If some clever algorithm could be found that uses future context without interference from glitches, then the prediction power of the LSTM could be very much increased with the bidirectional feature. In fact, the LSTM could possibly become unnecessary as a spline fit might work just as well. This could be the single biggest improvement to the algorithm; I just don't know if a solution exists.


### A discussion of other approaches

- I tried many different machine learning approaches while I worked on this. I settled on the LSTM for the following reasons:
  +  It is trained on only normal data and requires no prior knowledge of what the glitches look like. This is the biggest advantage. Combined with the simple glitch detection algorithm, an effective next point predictor can smooth over arbitrary shaped glitches as long as the first point triggers the outlier detector. 
  +  The machine has only one goal: to guess where the next point is. This is simple, and therefore it is less finicky than other approaches and easier to improve. 
  +  Compared to other next point predictors, like a simple CNN, the LSTM learns how to store information about the long-term trends in the signal, which is very appropriate for the decaying oscillations in EXAFS data. 
  +  Mentioned in future work, the LSTM approach is not far from being able to make much better predictions or from deglitching as the data come in.

- I found that the purely statistical method described in the paper "An algorithm for the automatic deglitching of x-ray absorption spectroscopy data" by S. M. Wallace *et al.* works decently, but I found it gave a fair number of false positives and false negatives. It also only works on point glitches.

- With so much hard-coding surrounding the machine learning model, I've wondered if decent results could come from simply fitting a spline to the chunks of data and extrapolating for the next prediction. Or using the chunk to estimate the first and possibly second or third derivative to use a Taylor series approximation for the prediction. I think it could be worth checking, and could avoid some issues with the transformation of the signal and the changes in sampling rates.

- I tried using a convolution neural network to take in chunks and classify them based on the glitch they contained, but it didn't seem to be as promising as the LSTM. It is discussed more in its own section below.

- The uneven sampling problem
  + The model works almost perfectly on data that is evenly sampled and has similar characteristics in shape to the data on which it was trained. In these circumstances the potential of this LSTM model to be a powerful tool is clear. The largest hinderance (I think) is the extreme variation in sampling rate in the data. The sampling rate problem noticeably affected the performance of the rolling CNN as well, and is apparently a well known problem in the field of machine learning in general. A model that interprets the data the way we do--as a curve instead of a list of values--could be immune to changes in sampling rates and therefore much more effective in its predictions. 
  + I tried a few ideas, such as getting the model to guess the slope between points, feeding it the derivatives, and feeding it the energy values with the mu values (and a couple different combinations of the three) and none worked better than just having the model guess the next point given all the previous points.
  + The paper "A review of irregular time series data handling with gated recurrent neural networks" by P. B. Weerakody *et al.* discusses some approaches to the irregularly spread data problem. I like the idea of using the LSTM to form the data into an ODE problem (quoted from the paper):
>Neural ODEs [83] are a family of continuous-time models which parameterize the derivative of the hidden state using deep neural networks. In these models, the hidden state is defined as the solution to an ODE initial-value problem, where the hidden state can be calculated at any time (t) using a numerical ODE solver. Unlike traditional RNN based models, these models develop a continuous-time function and are not bound to discrete-time sequences. RNN with ODEs uses the update function of a gated RNN model and ODE differential equation solver to produce RNN models as a function of continuous-time. The resulting RNN models are capable of learning the dynamics between observations and therefore, naturally handle arbitrary time gaps prevalent in irregular and sparse data.
  + The ODE approach, and others described in this paper, is where I would first look for more ideas about how to better address the uneven sampling problem. 


# 2. LSTM_on_mu.py

When run, this file trains and tests a new LSTM model for next point prediction and saves it as (something similar to) lstm_mu.pth. For operating the deglitcher, this file is only useful for the `LSTMModel()` class it contains, which defines the structure of the LSTM model that is used for loading the trained model into deglitch_mu.py. The script is just used for training the model, I think in-line comments are clear enough to describe what each parameter does if you want to fiddle with them.

The hyperparameters should all be close to optimal as there are in the script now. I've fiddled with them quite a bit, but have not embarked on serious systematic trials, so there could still be room for improvement. 


# 3. deglitch_mu_w_chi_transform.py, LSTM_on_chi1.py, and lstm_chi1.pth

These files function identically to deglitch_mu.py, LSTM_on_mu.py, and lstm_mu.pth, but uses a complete transformation of mu to chi before making predictions with the LSTM. deglitch_mu.py almost transforms to chi anyway, but has a few fewer restrictions. One of them being that the cropping point can be earlier than E_0.

deglitch_mu_w_chi_transform.py has not been updated to include the algorithms that handle the change in sampling rates. If you would like to use chi as the transformation for predictions, the chi transformation class can be copied or imported into the deglitch_mu.py script and `self.transform` can be changed in the `Mu_deglitcher` class. I haven't tried it, but I can't think of a reason for it not to work with that simple substitution. There may be issues with the x-axis now being k instead of E. 

# 4. deglitch_sigval_range.py

A simple script that contains a function to deglitch a signal repeatedly with different values for `sig_val`. The idea is that prominent glitches--and only prominent glitches--should should be identified by the algorithm for a range of values for `sig_val` but that range is not known beforehand. The function in this script identifies the unique set of point and step glitches that was reported more times in a range of consecutive values of `sig_val` than any other unique reported set of point and step glitches. In other words: as the function tries deglitching with increasing values of `sig_val`, there will be unique sets of point and step glitches that are reported many times in a row. The unique set that is reported the most times in a row is determined to be the best guess for the locations of real glitches. The function may also return the "best" deglitched spectrum. 

# 5. find_glitch_w_rolling_CNN.py, rolling_CNN_classifier.py, and rolling_glitch_classifier1.pth

These files are for the training and use of a convolutional neural network that takes in chunks of the sequence and classifies them as having a step glitch, monochromator glitch, or none. With such a classifier, it can be used to scan signals for the presence of the glitches and determine the range of point in which they are present. These files are not as well documented as the others.

A rolling CNN glitch classifier requires training on normal and labelled glitchy data, and I found it to be less reliable than the LSTM for finding glitches of different sizes and slightly different shapes. There is no easy way to tune the sensitivity of it, and it can be inconsistent about which point exactly it thinks is the glitchy one. However, a rolling CNN has the advantage that it *only* needs to look at the chunk of points, which means the shape of the signal as a whole does not really affect its ability to detect glitches. And as it scans the data, it has the opportunity to detect the same glitch multiple times as a way of double-checking itself. It may also be able to look for and identify glitches of different shapes, like monochromator glitches. For a completely different approach to the LSTM, this would be my second choice. 


# 6. discriminate_pixels.py

Defines functions that take a set of fluorescence scans from the independent detectors (pixels) and sort out the ones without useful signal. Running the script runs a random example on sets of real data.

First, for each signal in the set of measurements, the list of differences between consecutive points is extracted. Then the mean and standard deviation for each list of consecutive differences is calculated. By using the mean and standard deviation of the consecutive differences, each signal is reduced to a two-dimensional space. It was found that this method of dimensionality reduction resolves the useful scans into an isolated cluster. DBSCAN is used to find the cluster of scans that are all most similar to each other and reject the outliers. The functions in this file return the indices of the pixel scans that belong to the (or a) cluster.

### Usage

Run the function `find_good_pix()` on a set of pixel scans. The inline comments are detailed. Follow the example. There are options for visualizing the process and the results.

### Notes

- The default parameters work fine for the data I have used for testing. min_samples should be kept above three, as there are usually three pixels that are zero or contain nans and they could be identified as a cluster together. If min_samples is too high, it may be that no cluster is identified. eps is the main tuning parameter. The higher eps is, the more points will be included in a cluster. The lower it is, the fewer points will be clustered, and the smaller clusters will be. Sometimes with a low eps, more than one cluster may be identified. This may be desirable if there are two meaningfully different types of scans. 

- In the dataset I have been using for testing, one pixel scan always contains a full list of nans. During pixel normalization, if a scan contains a nan, it is set to zeros. In every case I have tested, the zeroed scans are never part of the main cluster, but that might not always be true. In which case, you may end up trying to do analysis with nan values. This may be fixed in a future version where the labels for the zero and nan scans are set to -1 from the outset. 

- A future version may be made where the tuning of eps is interactive and the script displays the clustering plot for different values of eps.
