################################################################################
#  TLGProb: Two-Layer Gaussian Process Regression Model For
#           Winning Probability Calculation of Two-Team Sports
#  Github: https://github.com/MaxInGaussian/TLGProb
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import sys
sys.path.append("../")
from models import basketball_model


bas = basketball_model(
    database_path="database/",
    model_path="trained_gpr/")
bas.load_data()
while(1):
    bas.train_winning_team_model()
    bas.train_player_models()
