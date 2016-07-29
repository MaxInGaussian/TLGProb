"""
Created on Tue Sep 17 12:10:19 2015
@author: Max W. Y. Lam
"""
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
