################################################################################
#  TLGProb: Two-Layer Gaussian Process Regression Model For
#           Winning Probability Calculation of Two-Team Sports
#  Github: https://github.com/MaxInGaussian/TLGProb
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import sys
sys.path.append("../")
import pickle
from TLGProb import TLGProb, WrappedPredictor


TLGProb_NBA = TLGProb(
    database_path="database/",
    model_path="trained_models/")
TLGProb_NBA.load_data()
regression_methods = [
    "KernelRidge", "DecisionTreeRegressor", "AdaBoostDecisionTreeRegressor",
    "GradientBoostingRegressor", "RandomForestRegressor", "SSGPR"]
for regression_method in regression_methods:
    TLGProb_NBA.eval_accuracy(2015, threshold=0.5, regression_methods=["KernelRidge", regression_method])