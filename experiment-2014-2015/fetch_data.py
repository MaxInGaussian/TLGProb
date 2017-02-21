################################################################################
#  TLGProb: Two-Layer Gaussian Process Regression Model For
#           Winning Probability Calculation of Two-Team Sports
#  Github: https://github.com/MaxInGaussian/TLGProb
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

try:
    from TLGProb import NBA_Bot
except:
    print("TLGProb is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../")
    from TLGProb import NBA_Bot
    print("done.")


bot = NBA_Bot()
bot.get_all_games_by_year(2016, update=True)
bot.get_basketball_player_by_year(2016, update=True)
bot.get_basketball_game_log_by_year(2016, update=True)
