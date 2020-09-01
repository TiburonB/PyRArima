#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.
#
# The input directory input_dir (e.g. sample_data/) contains the dataset,
# training/
#   training0.csv
#   training1.csv
#   training2.csv
# evaluation/
#   evaluation0.csv
#   evaluation1.csv
#
# The output directory output_dir (e.g. sample_result_submission/) 
# will receive the predicted values (no subdirectories):
# 	predictions.csv
#
# The code directory submission_program_dir (e.g. sample_code_submission/) should contain your 
# code submission model.py (an possibly other functions it depends upon).
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, UPSUD, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 
#
# Main contributor: Isabelle Guyon May 2019

# =============================================================================
# ========================= BEGIN USER OPTIONS ================================
# =============================================================================
# Verbose mode: 
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
############## 
# 0: run the code normally 
# 1: be little more verbose (must set verbose=True)
# 2: write chearing results
# 3: stop before loading data
# 4: just list the directories and program version
debug_mode = 0
show_graphics = False

# Save your model
#################
save_model = False

# I/O defaults
##############
# If true, the previous output directory is not overwritten, it changes name
save_previous_results = False
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "../"
default_input_dir = root_dir + "sample_data"
default_output_dir = root_dir + "sample_results"
default_program_dir = root_dir + "ingestion_program"
default_submission_dir = root_dir + "sample_code_submission"

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# =========================== BEGIN MAIN PROGRAM ================================
if __name__=="__main__" and debug_mode<4:	
    #### Version of this ingestion program
    version = 1

    #### General purpose functions
    import time
    start_time = time.time()         # <== Mark starting time
    import os
    from sys import argv, path
    import datetime
    the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    #### Check whether everything went well (no time exceeded)
    execution_success = True
    
    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(argv[4])
    if verbose: 
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)
        print("Using submission_dir: " + submission_dir)

	# Our libraries
    path.append (program_dir)
    path.append (submission_dir)
    import data_io                       # general purpose input/output functions
    from data_io import vprint           # print only in verbose mode
    from data_manager import DataManager # load/save data and get info about them
    from model import Model    			 # example model, in scikit-learn style

    if debug_mode >= 4: # Show library version and directory structure
        data_io.show_dir(".")
        
    # Move old results and create a new output directory (useful if you run locally)
    if save_previous_results:
        data_io.mvdir(output_dir, output_dir+'_'+the_date) 
    data_io.mkdir(output_dir) 
        
    vprint( verbose,  "\n========== Ingestion program version " + str(version) + " ==========\n") 
    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_version()
        data_io.show_io(input_dir, output_dir)
        exit(0)
          
    vprint( verbose,  "****************************************************")
    vprint( verbose,  "******** Processing spatio-temporal dataset ********")
    vprint( verbose,  "****************************************************")

    #### Instanciate input data manager and load data
    vprint( verbose,  "========= Reading and converting data ==========")
    Din = DataManager(datatype="input", verbose=verbose) 
    Din.loadData(input_dir)
    vprint( verbose, Din)
    vprint( verbose,  "[+] Size of uploaded data  {:5.2f} bytes".format(data_io.total_size(Din)))
    
    #### Instanciate output data manager and load data
    Dout = DataManager(datatype="output", verbose=verbose)
    Dout.col_names = Din.col_names[Din.ycol0:]
    Dout.horizon = Din.horizon
    Dout.stride = Din.horizon
    
    #### In debug mode, cheat and get the solution too
    if debug_mode>1:
        Dsol = DataManager(datatype="input", verbose=verbose) 
        Dsol.loadData(input_dir)
    
    #### Instanciate predictive model
    vprint( verbose,  "======== Creating model ==========")
    M = Model()
    #### THIS IS YOUR STUFF!!
    
    #### MAIN LOOP OVER DATA SAMPLES: 
    vprint( verbose,  "======== Making predictions =====")
    X, tx = Din.getHistoricalData()
    ty = Din.t[Din.now-Din.stride:Din.now-Din.stride+Din.horizon] 
    while ty.shape[0]!=0:
        # Train or adapt with warm start (does not necessarily do anything)
        M.train(X, tx) 
        # Make predictions (the same X matrix is used)
        if debug_mode>1:
            Y, ty = Dsol.getFutureOutcome() # Cheating predictions = truth values (for debug purposes)
        else:
            Y = M.predict(X, num_predicted_frames=Din.horizon, ycol0=Din.ycol0)
            Y = Y[:len(ty),:] # Clip predictions for ground truth we do not have
        Dout.appendData(Y, ty) 
        if debug_mode>0:
            vprint( verbose, ty)
            vprint( verbose, Dout)
            vprint( verbose, Dout.X.shape)
            vprint( verbose, Dout.t.shape)
            vprint( verbose,  "----------------")
        X, tx = Din.getHistoricalData()
        ty = Din.t[Din.now-Din.stride:Din.now-Din.stride+Din.horizon]  
        
    # All predictions:
    vprint( verbose, Dout)
    #### In this setting, we do not save the model, but this could be done
    #### It would not make sense to reload a model that already saw all the data
    if save_model:
        M.save(path=output_dir)     
    
    #### Saving the results        
    Dout.saveData(os.path.join(output_dir, "prediction")) 
    if debug_mode>0:
        # Save results also as CSV for visual inspection
        Dout.saveData(os.path.join(output_dir, "prediction"), format="csv")
    if show_graphics:
        Dout.show(transpose=False)   
        
    vprint( verbose,  "[+] End process, time spend {:5.2f} sec".format(time.time()-start_time))
