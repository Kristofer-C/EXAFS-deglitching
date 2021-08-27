# -*- coding: utf-8 -*-
"""
Trying to deglitch a spectrum with the LSTM method by running the deglitching
function with a range of significance values. 
"""

from deglitch_LSTM_chi import *

def deglitch_range(glitchy,
                   model=None,
                   chunk_size=10,
                   out_size=1,
                   return_deglitched=False):
    """
    Scan through a range of significance values, and pick the list of step
    glitch guesses and point glitch guesses that appeared the most times in
    a row. 
    """

    # Keep track of the number of times the current list has appeared in a row
    count=0
    # Keep track of the previous set of guesses
    point_prev=[]
    step_prev=[]
    # The most times so far that an exact list has appeared in a row
    max_consec=0
    
    # Scan through a list of values
    for sig_val in np.logspace(-3.0, -2, 10):    
        
        # Perform the deglitching algorithm. A lot of the predicitions are 
        # probably redundant, but depending on how it goes through and performs
        # the deglitching, it will change further predictions
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
        
        # If this list of guesses is the same as the last one:
        if point_glitch_guesses==point_prev and step_glitch_guesses==step_prev:
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
        
    if return_deglitched:
        return best_points, best_steps, best_deglitched
    else:    
        return best_points, best_steps
    
    


if __name__=="__main__":
    
    # Load the trained model
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    chunk_size=16
    hidden_size=32
    out_size=1
    batch_size=1
    num_layers=4
    bidirectional=False
    drop_prob=0.0
    model=LSTMModel(chunk_size, 
                    hidden_size, 
                    out_size, 
                    batch_size,
                    num_layers, 
                    drop_prob,
                    bidirectional,
                    device).to(device)
    model.load_state_dict(torch.load("lstm_chi1.pth", map_location=torch.device(device)))
    model.init_hidden(batch_size)
    
    
    # Create places for the original signal, the one with added glithces, 
    # the deglitched signal, and the list of predicted points
    
    chi_dir="chi_test"
    chilist=os.listdir(chi_dir)
    chiname=np.random.choice(chilist)
    chi=np.genfromtxt(os.path.join(chi_dir, chiname))
    clean=chi/(max(chi)-min(chi))
    x=np.arange(len(chi))
    
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
        
    plt.plot(x,glitchy)
    
    
    point_guesses, step_guesses=deglitch_range(glitchy,
                                               model, 
                                               chunk_size,
                                               out_size)
    
    print(point_guesses, step_guesses)
