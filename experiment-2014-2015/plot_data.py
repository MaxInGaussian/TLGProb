"""
Created on Tue Mar 25 01:11:39 2016
@author: Max W. Y. Lam
"""
import sys
sys.path.append("../")
import matplotlib.pyplot as plt
from bisect import bisect_left
from models import basketball_model
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.dates import MONDAY
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter


def plot_unsmoothed_feature(bas, fig, ax, line, feature, player=None):
    mondays = WeekdayLocator(MONDAY)
    months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
    monthsFmt = DateFormatter("%b '%y")
    feature_ind = bas.key_to_player_input_id[feature]
    lines = []
    dates = []
    for y, m, d in bas.game_dict['date']:
        date = dt.date(y, m, d)
        if(date not in dates):
            dates.append(date)
    if(player is not None):
        line_x, line_y = [], []
        for i, date_int in enumerate(bas.player_to_date[player]):
            date = dt.date(int(date_int/10000.), int(date_int/100.)%100,date_int%100)
            line_x.append(date)
            line_y.append(bas.player_to_attributes[player]["performance"][i, feature_ind])
        ax.plot_date(line_x, line_y, line, label=feature.upper()+" performance")
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        ax.autoscale_view()
        ax.grid(True)
    else:
        for player in bas.all_player:
            line_x, line_y = [], []
            for i, date_int in enumerate(bas.player_to_date[player]):
                date = dt.date(int(date_int/10000.), int(date_int/100.)%100,date_int%100)
                line_x.append(date)
                line_y.append(bas.player_to_attributes[player]["performance"][i, feature_ind])
            ax.plot_date(line_x, line_y, line, label=feature.upper()+" performance")
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(monthsFmt)
            ax.xaxis.set_minor_locator(mondays)
            ax.autoscale_view()
            ax.grid(True)
    fig.autofmt_xdate()

def plot_smoothed_feature(bas, fig, ax, line, feature, player=None):
    mondays = WeekdayLocator(MONDAY)
    months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
    monthsFmt = DateFormatter("%b '%y")
    feature_ind = bas.key_to_player_input_id[feature]
    lines = []
    dates = []
    for y, m, d in bas.game_dict['date']:
        date = dt.date(y, m, d)
        if(date not in dates):
            dates.append(date)
    if(player is not None):
        line_x, line_y = [], []
        for i, date_int in enumerate(bas.player_to_date[player]):
            date = dt.date(int(date_int/10000.), int(date_int/100.)%100,date_int%100)
            line_x.append(date)
            line_y.append(bas.player_to_attributes[player]["smoothed_performance"][i, feature_ind])
        ax.plot_date(line_x, line_y, line, label=feature.upper()+" ability")
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        ax.autoscale_view()
        ax.grid(True)
    else:
        for player in bas.all_player:
            line_x, line_y = [], []
            for i, date_int in enumerate(bas.player_to_date[player]):
                date = dt.date(int(date_int/10000.), int(date_int/100.)%100,date_int%100)
                line_x.append(date)
                line_y.append(bas.player_to_attributes[player]["smoothed_performance"][i, feature_ind])
            ax.plot_date(line_x, line_y, line, label=feature.upper()+" ability")
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(monthsFmt)
            ax.xaxis.set_minor_locator(mondays)
            ax.autoscale_view()
            ax.grid(True)
    fig.autofmt_xdate()

def plot_players_performance(bas):
    plt.figure(len(bas.key_to_player_input_id)*2)
    lines = []
    dates = []
    for y, m, d in bas.game_dict['date']:
        date = y*10000+m*100+d
        if(date not in dates):
            dates.append(date)
    for player in bas.all_player:
        line_x, line_y = [], []
        for i, date in enumerate(bas.player_to_date[player]):
            date_ind = bisect_left(dates, date)
            line_x.append(date_ind)
            line_y.append(bas.player_to_attributes[player]["+/-"][i])
        plt.plot(line_x, line_y)
    plt.title("Time Series of Players' Performances")
    plt.xlabel("Date Index")
    plt.ylabel("Player Performance")

def plot_players_ability(bas):
    plt.figure(len(bas.key_to_player_input_id)*2+1)
    lines = []
    dates = []
    for y, m, d in bas.game_dict['date']:
        date = y*10000+m*100+d
        if(date not in dates):
            dates.append(date)
    for player in bas.all_player:
        line_x, line_y = [], []
        for i, date in enumerate(bas.player_to_date[player]):
            date_ind = bisect_left(dates, date)
            line_x.append(date_ind)
            line_y.append(bas.player_to_attributes[player]["smoothed_plus_minus"][i])
        plt.plot(line_x, line_y)
    plt.title("Time Series of Players' Abilities")
    plt.xlabel("Date Index")
    plt.ylabel("Player Ability")

bas = basketball_model()
bas.load_data()
fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
player = "LeBron James"
feature = "3p"
plot_unsmoothed_feature(bas, fig, ax, "b--", feature, player)
plot_smoothed_feature(bas, fig, ax, "r-", feature, player)
# plt.title("Time Series Plot of "+player+"'s "+feature.upper())
plt.ylabel(feature.upper())
plt.legend(loc="upper center")
plt.tight_layout()
plt.show()