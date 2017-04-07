################################################################################
#  TLGProb: Two-Layer Gaussian Process Regression Model For
#           Winning Probability Calculation of Two-Team Sports
#  Github: https://github.com/MaxInGaussian/TLGProb
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import sys, os
import csv
import numpy as np
import random
from SSGP import SSGP

class WrappedPredictor(object):
    
    def __init__(self, regression_method="", predictor=None, X_train=None, y_train=None):
        self.regression_method = regression_method
        if(predictor is None):
            self.predictor = SSGP()
        else:
            self.predictor = predictor
        if(self.regression_method == "SSGPR"):
            self.hashed_name = self.predictor.hashed_name
        else:
            self.hashed_name = random.choice("ABCDEF")+str(hash(self)&0xffff)
            self.X_train = X_train
            self.y_train = y_train
        
    def predict(self, X_test, y_test=None):
        if(self.regression_method == "SSGPR"):
            return self.predictor.predict(X_test, y_test)
        mu = self.predictor.predict(X_test)
        std = np.sqrt(np.mean((self.predictor.predict(self.X_train)-self.y_train)**2))
        if(y_test is None):
            return mu, std
        y_mu = np.mean(y_test.ravel())
        self.mse = np.mean(np.square(y_test-mu))
        self.nmse = np.mean(np.square(y_test-mu))/np.mean(np.square(y_test-y_mu))
        self.mnlp = np.mean(np.square((y_test-mu)/std)+2*np.log(std))
        self.mnlp = 0.5*(np.log(2*np.pi)+self.mnlp)
        return self.mse, self.nmse, self.mnlp
    
    def save(self, path):
        if(self.regression_method == "SSGPR"):
            self.predictor.save(path)
            return;
        import pickle
        with open(path, "wb") as save_f:
            pickle.dump([self.predictor, self.X_train, self.y_train, self.hashed_name],
                save_f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        if(self.regression_method == "SSGPR"):
            self.predictor.load(path)
            return;
        import pickle
        with open(path, "rb") as load_f:
            self.predictor, self.X_train, self.y_train, self.hashed_name = pickle.load(load_f)
        