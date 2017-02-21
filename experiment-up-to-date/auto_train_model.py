"""
Created on Tue Sep 17 12:10:19 2015
@author: Max W. Y. Lam
"""
import sys
sys.path.append("../")
from models import basketball_model


while(1):
    bas = basketball_model()
    bas.load_data()
    bas.train_winning_team_model()
    bas.train_player_models()
