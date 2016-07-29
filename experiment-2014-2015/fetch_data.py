"""
Created on Tue Sep 15 17:11:39 2015
@author: Max W. Y. Lam
"""
import bots


bot = bots.basketball_bot()
bot.get_all_games_by_year(2015, update=True)
bot.get_basketball_player_by_year(2015, update=True)
bot.get_basketball_game_log_by_year(2015, update=True)
