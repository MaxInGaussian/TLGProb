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
if(TLGProb_NBA.winning_team_model_dataset is None):
    TLGProb_NBA.get_winning_team_model_dataset()
dataset = TLGProb_NBA.winning_team_model_dataset
X, y = dataset
regression_methods = [
    "KernelRidge", "DecisionTreeRegressor", 
    "AdaBoostDecisionTreeRegressor", "GradientBoostingRegressor",
    "RandomForestRegressor", "SSGPR"
]
regression_performances = {reg:None for reg in regression_methods}
for regression_method in regression_methods:
    model_path = TLGProb_NBA.get_winning_team_model_path(regression_method)
    model = WrappedPredictor(regression_method)
    model.load(model_path)
    regression_performances[regression_method] = model.predict(X, y)