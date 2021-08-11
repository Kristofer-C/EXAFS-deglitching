# EXAFS-deglitching

# 1. deglitch_mu.py

Defines the classes used to deglitch mu and provides an example.  

The deglitching algorithm uses a trained Long Short-Term Memory (LSTM) nerual network to make predictions on successive points in the sequence. If the next measured value is statistically significantly far from the value predicted for it, then that point is labelled as a glitch. The glitchy point is replaced with the predicted value to remove the glitch. If the following point is also determined to be a glitch, then it is determined to be the second point in a step glitch, and the rest of the signal is shifted by the difference between the first predicted point and the corresponding measured value.

The LSTM has a hard time making predictions on mu as it is, so mu is transformed before deglitching by rescaling it to be between zero and one, and then subtracting a fitted spline. This has the effect of centering the oscillations around zero, much like chi, and stabilizes the predictions made by the LSTM. The signal is deglitched in this form and then transformed back with the orginal transformation parameters.

### Usage

Define an object with the Mu_Deglitcher class. Use one of .run() or .run_twostage() with their corresponding arguments and parameters to return the deglitched signal. Inline comments are detailed. Follow the example given.

### Dependencies

Imports LSTMModel from LSTM_on_mu. This is the class that defines the model used for making predictions. When loading the model, deglitch_mu.py uses the file lstm_mu.pth to load all the trained weights and biases. The hyperparamters for the model are defined in the Mu_Deglitcher class and can not be changed for the model to load. 

### Notes

It is recommended to use .run_twostage() because it more robustly detects and fixes glitches, particularly when large step glitches are present.

The algorithms are designed to remove point and step glitches. Removing point glitches is usually very successful, and false positives are typically inconsequential. False positive step glitches can corrupt the signal more, so it is worth double-checking the necessity when a step glitch is reported. Monochromator glitches involve more than one deviant point and the model is not designed to identify or remove them. However, some initial testing has shown that a monochromator glitch can end up being treated as a combination of point and step glitches and may still be removed entirely. A good indication of a monochromator glitch in the signal is several step glitches being reported in consecutive points. 

The indeces of the detected glitches are not returned, and are only printed out if visualize=True. It may be desired to have the detected glitch indeces reported every time with the deglitched signal. Let me know.

Cropping off some initial part of the measured mu is not strictly necessary. The deglitching algorithm runs mostly fine with crop_ind=0. However, the sharp increases in energy sampling, the transition to the edge jump, and the pre-edge peaks can all cause false positive identifications of step glitches. For best results, crop off the measured points until just after the pre-edge peak. This may be earlier than the e0 used to transform mu to chi, but it may end up being convenient to just use e0 as the cropping point. In a future version, the choice of cropping point may be automated.

The machine learning algorithm is trained on an energy sampling rate that loosely matches that of the highest sampling rate present in the data I used for testing. The algorithm works decently with different sampling rates, but in areas with strong oscillations, the next point predictions can drift from the real data in different ways. If the measured data contains a higher sampling rate than what the model was trained on, then the model will tend to make predictions that are closer to zero than the real data. This is because the model expects each oscillation to take a fewer number of points, so it guesses that the signal should be returning to zero sooner than it is. If the measured data contains lower sampling rates than what the model was trained on, then the predictions will tend to lag behind the real data because the model is accustomed to guessing smaller changes between points. The result is that poor predictions can cause less consistent glitch detection and less effective glitch fixing.

The changing sampling rates are not too problematic for the data used for testing, but for robust implementation, something in a future verion should account for the different sampling rates. Some form of upsampling may be useful.

# 2. LSTM_on_mu.py


# 3. deglitch_mu_w_chi_transform.py

Functions identically to deglitch_mu.py, but uses a complete transformation of mu to chi before making predictions with the LSTM. deglitch_mu.py almost transforms to chi anyway, but has a few fewer restrictions. One of them being that the cropping point can be earlier than e0.


# 4. descriminate_pixels.py
