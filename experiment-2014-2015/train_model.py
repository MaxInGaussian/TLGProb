################################################################################
#  TLGProb: Two-Layer Gaussian Process Regression Model For
#           Winning Probability Calculation of Two-Team Sports
#  Github: https://github.com/MaxInGaussian/TLGProb
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import sys
sys.path.append("../")
from TLGProb import TLGProb


TLGProb_NBA = TLGProb(
    database_path="database/",
    model_path="trained_models/")
TLGProb_NBA.load_data()
regression_methods = [
    "KernelRidge", "DecisionTreeRegressor", 
    "AdaBoostDecisionTreeRegressor", "GradientBoostingRegressor",
    "RandomForestRegressor", "SSGPR"
]
for regression_method in regression_methods:
    TLGProb_NBA.train_player_models(regression_method=regression_method)
    TLGProb_NBA.train_winning_team_model(regression_method=regression_method)