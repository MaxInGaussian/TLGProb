"""
Created on Tue Sep 15 17:11:39 2015
@author: Max W. Y. Lam
"""
import sys
sys.path.append("../")
from datetime import datetime
from models import basketball_model

while(True):
    bas = basketball_model()
    bas.load_data()
    res = bas.generate_next_prediction()
    home_dir = os.path.dirname(os.path.realpath('__file__'))
    f = open(os.path.join(home_dir, "NBA_Online_Prediction.txt"), "w")
    f.write(res)
    f.close()
    print(res)
    time.sleep(3600)

