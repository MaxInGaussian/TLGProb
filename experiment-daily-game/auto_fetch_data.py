import models
import bots
from datetime import datetime
import os
import sys
import time


bot = bots.basketball_bot()
while(True):
    month = datetime.today().month
    year = datetime.today().year + (1 if month>8 else 0)
    bot.get_basketball_player_by_year(year, update=True)
    bot.get_basketball_game_log_by_year(year, update=True)
    bot.get_all_games_by_year(year, update=True)
    time.sleep(3600)
