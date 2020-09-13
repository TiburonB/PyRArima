import pickle
import numpy as np 				# We assume that numpy arrays will be used
import os
from data_io import vprint
import time
import random

# python3 ingestion_program/ingestion.py public_data sample_results ingestion_program sample_code_submission

###############################################################################################################
# need rpy2 to call R from python
#import rpy2
#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr
################################################################################################################

class Model():
    # change verbose to False before submitting!!!!!!!!!!!!
    def __init__(self, hyper_param=[], path="", verbose=True):
        ''' Define whatever data member you need (model paramaters and hyper-parameters).
        hyper_param is a tuple.
        path specifies the directory where models are saved/loaded.'''
        self.version="Persitent"
        self.hyper_param=hyper_param
        self.model_dir = path
        self.verbose = verbose
        vprint(self.verbose, "Version = " + self.version)

    def train(self, Xtrain, Ttrain=[]):
        '''  Adjust parameters with training data.
        May be called several times with increasingly more data or new data.
        Consider doing a "warm start".
        Xtrain is a matrix of frames (frames in lines, features/variables in columns)
        Ttrain is the optional time index. The index may not be continuous (e.g. jumps or resets)
        Typically Xtrain has thousands of lines.''' 
        vprint(self.verbose, "Model :: ========= Training model =========")
        start = time.time()
        # Do something
        end = time.time()
        vprint(self.verbose, "[+] Success, model trained in %5.2f sec" % (end - start))
         
    def predict(self, Xtest, num_predicted_frames=8, ycol0=0):
        ''' Make predictions of the next num_predicted_frames frames.
        Start at variable ycol0 only (do not predict the values of the first
        0 to ycol0-1 variables).
        For this example we predict persistence of the last frame.'''
        vprint(self.verbose, "Model :: ========= Making predictions =========")
        vprint(self.verbose, "===============================================")
        start = time.time()
        #Ytest = np.array([Xtest[random.randint(0,10),ycol0:]] * num_predicted_frames) 
        
        
        ######################        
        # import rpy2's package module
        import rpy2
        import rpy2.robjects as robjects
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.packages import importr

        # import R's "base" package
        base = rpackages.importr('base')


        # import R's utility package
        utils = rpackages.importr('utils')

        # select a mirror for R packages
        utils.chooseCRANmirror(ind=1) # select the first mirror in the list

        if rpy2.robjects.packages.isinstalled('forecast', lib_loc = rpy2.__path__[0]) == False:
            utils.install_packages('forecast', lib = rpy2.__path__[0])
        forecast = importr('forecast', lib_loc = rpy2.__path__[0])

        ts = robjects.r('ts')
        
        #from rpy2.robjects.vectors import FloatVector
        #from rpy2.robjects.vectors import IntVector
        #from rpy2.robjects.vectors import BoolVector

        #from rpy2.robjects import pandas2ri
        
        from rpy2.robjects import pandas2ri
        from rpy2.robjects import vectors

        pandas2ri.activate()
        ######################        
        
        
        Ytest = np.zeros((7,57))
        
        # Code assumes daily data (not aggregated. Arima will break if it's run on aggregated data. 
        # I've provided commented code that should undo aggrgation in inputs into model and redo
        # aggregation to return the predictions (Ytest)

        # undo aggregation:
        future_starts = []
        for col in range(ycol0, Xtest.shape[1]):
            init = Xtest[0,col]
            for row in range(1, Xtest.shape[0]):
                Xtest[row,col] -= init
                init += Xtest[row,col]
            future_starts.append(init)

        for col in range(ycol0, Xtest.shape[1]):
            #print(col)
            dtp = num_predicted_frames - 1     # days to predict
            ndpat = num_predicted_frames   # number days to predict at a time
            dat = Xtest[1:,col]
            #print(dat)
            #print(len(dat))
            sum_RMSE = 0
            f = ts(dat, frequency = 1, start = 1, end = len(dat))
            best_params = robjects.IntVector([0,0,0])
            best_RMSE = 1000000
        
            
            for p in range(1,5) :
                for q in range(0,5) :
                    for d in range(0,3) :
                        try: 
                            t_order = robjects.IntVector([p,d,q])
                            fit2 = forecast.Arima(f, order = t_order, xreg = robjects.r("NULL"), include_mean = True, include_drift = False, biasadj = False, method = "ML", model = robjects.r("NULL"))
                            RMSE = forecast.accuracy(fit2)[0][2] #RMSE          
                            if  RMSE < best_RMSE  :
                                best_RMSE = RMSE
                                best_params = robjects.IntVector([p,d,q])
                        except:
                            continue


            best_opts = robjects.BoolVector([True, False])
            possible_opts = robjects.BoolVector([True, False])
            for  mean_opt in range(0,1) :
                for  drift_opt in range(0,1) :
                    mean_opt = possible_opts[mean_opt]
                    drift_opt = possible_opts[drift_opt]
                    fit2 = forecast.Arima( f, order = best_params, xreg = robjects.r("NULL"), include_mean = mean_opt, include_drift = drift_opt, biasadj = False, method = "ML", model = robjects.r("NULL"))
                    RMSE = forecast.accuracy(fit2)[0][2] #RMSE
                    if ( RMSE  < best_RMSE ) :
                            #print(paste("Reset best_params to (p,d,q) = (", p, ",", d, ",", q , ")", sep = ""))
                            best_RMSE = RMSE
                            best_opts = robjects.BoolVector([mean_opt, drift_opt])

            #print("best params = ", best_params)
            #print("best opts = ", best_opts)
            fit2 = forecast.Arima( f, order = best_params, xreg = robjects.r("NULL"), include_mean = best_opts[0], include_drift = best_opts[1], biasadj = False, method = "ML", model = robjects.r("NULL"))
        #    print(forecast.forecast(fit2, ndpat))
        #    print(forecast.forecast(fit2, ndpat)[0])
        #    print(forecast.forecast(fit2, ndpat)[1])
        #    print(forecast.forecast(fit2, ndpat)[2])
        #    print(forecast.forecast(fit2, ndpat)[3])
            Ytest[:,col] = forecast.forecast(fit2, ndpat)[3]
            #print(Ytest)

        #print(Xtest.shape)    # (78, 57)
        #print(Xtest.shape[0]) # 78
        #print(Ytest.shape)    # typically (7, 57)

        # reconstruct aggregated predictions
        for col in range(ycol0, Xtest.shape[1]):
            init = future_starts[col]
            for row in range (0, num_predicted_frames - 1):
                tinc = init
                init += Ytest[row,col]
                Ytest[row,col] += tinc


        end = time.time()
        vprint(self.verbose, "[+] Success, predictions made in %5.2f sec" % (end - start))
        vprint(self.verbose, "Model :: ======== Predictions finished ========")
        return Ytest
        
    def save(self, path=""):
        ''' Save model.'''
        if not path: path = self.model_dir
        vprint(self.verbose, "Model :: ========= Saving model to " + path)
        pickle.dump(self, open(os.path.join(path, '_model.pickle'), "w"))

    def load(self, path=""):
        ''' Reload model.'''
        if not path: path = self.model_dir
        vprint(self.verbose, "Model :: ========= Loading model from " + path)
        self = pickle.load(open(os.path.join(path, '_model.pickle'), "w"))
        return self
