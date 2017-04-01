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
TLGProb_NBA.train_player_models(regression_method="RidgeCV")
TLGProb_NBA.train_winning_team_model(regression_method="RidgeCV")
TLGProb_NBA.eval_accuracy(2015)