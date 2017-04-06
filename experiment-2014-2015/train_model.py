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
    "SSGPR", "KNeighborsRegressor",
    "DecisionTreeRegressor", "AdaBoostDecisionTreeRegressor",
    "GradientBoostingRegressor", "RandomForestRegressor"]
for regression_method in regression_methods:
    model_path = get_player_model_path_by_pos(regression_method=regression_method)
    with open(model_path, "rb") as load_f:
        model = pickle.load(load_f)
        model = WrappedPredictor(regression_method, model, X, y)
        mse, nmse, mnlp = model.predict(X.copy(), y.copy())
        print("RESULT OF %s MODEL (%s):" % (regression_method, model.hashed_name))
        print("\tMSE = %.5f\n\tNMSE = %.5f\n\tMNLP = %.5f"%(mse, nmse, mnlp))
        model.save(model_path)
for regression_method_1 in regression_methods:
    for regression_method_2 in regression_methods:
        TLGProb_NBA.eval_accuracy(2015, regression_methods=[regression_method_1, regression_method_2])