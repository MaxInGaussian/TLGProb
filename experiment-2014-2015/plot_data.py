################################################################################
#  TLGProb: Two-Layer Gaussian Process Regression Model For
#           Winning Probability Calculation of Two-Team Sports
#  Github: https://github.com/MaxInGaussian/TLGProb
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.dates import MONDAY
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
try:
    from TLGProb import TLGProb
except:
    print("TLGProb is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../")
    from TLGProb import TLGProb
    print("done.")


def plot_unsmoothed_feature(TLGProb_NBA, fig, ax, line, feature, player=None):
    mondays = WeekdayLocator(MONDAY)
    months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
    monthsFmt = DateFormatter('%Y-%m')
    feature_ind = TLGProb_NBA.key_to_player_input_id[feature]
    lines = []
    dates = []
    for y, m, d in TLGProb_NBA.game_dict['date']:
        date = dt.date(y, m, d)
        if(date not in dates):
            dates.append(date)
    if(player is not None):
        line_x, line_y = [], []
        for i, date_int in enumerate(TLGProb_NBA.player_to_date[player]):
            date = dt.date(int(date_int/10000.), int(date_int/100.)%100,date_int%100)
            line_x.append(date)
            line_y.append(TLGProb_NBA.player_to_attributes[player]["performance"][i, feature_ind])
        ax.plot_date(line_x, line_y, line, label=feature.upper()+" performance")
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        ax.autoscale_view()
        ax.set_ylim([np.min(line_y)-0.1*np.std(line_y), np.max(line_y)+0.6*np.std(line_y)])
    else:
        for player in TLGProb_NBA.all_player:
            line_x, line_y = [], []
            for i, date_int in enumerate(TLGProb_NBA.player_to_date[player]):
                date = dt.date(int(date_int/10000.), int(date_int/100.)%100,date_int%100)
                line_x.append(date)
                line_y.append(TLGProb_NBA.player_to_attributes[player]["performance"][i, feature_ind])
            ax.plot_date(line_x, line_y, line, label=feature.upper()+" performance")
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(monthsFmt)
            ax.autoscale_view()
    fig.autofmt_xdate()

def plot_smoothed_feature(TLGProb_NBA, fig, ax, line, feature, player=None):
    mondays = WeekdayLocator(MONDAY)
    months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
    monthsFmt = DateFormatter('%Y-%m')
    feature_ind = TLGProb_NBA.key_to_player_input_id[feature]
    lines = []
    dates = []
    for y, m, d in TLGProb_NBA.game_dict['date']:
        date = dt.date(y, m, d)
        if(date not in dates):
            dates.append(date)
    if(player is not None):
        line_x, line_y = [], []
        for i, date_int in enumerate(TLGProb_NBA.player_to_date[player]):
            date = dt.date(int(date_int/10000.), int(date_int/100.)%100,date_int%100)
            line_x.append(date)
            line_y.append(TLGProb_NBA.player_to_attributes[player]["smoothed_performance"][i, feature_ind])
        ax.plot_date(line_x, line_y, line, label=feature.upper()+" ability")
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        ax.autoscale_view()
    else:
        for player in TLGProb_NBA.all_player:
            line_x, line_y = [], []
            for i, date_int in enumerate(TLGProb_NBA.player_to_date[player]):
                date = dt.date(int(date_int/10000.), int(date_int/100.)%100,date_int%100)
                line_x.append(date)
                line_y.append(TLGProb_NBA.player_to_attributes[player]["smoothed_performance"][i, feature_ind])
            ax.plot_date(line_x, line_y, line, label=feature.upper()+" ability")
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(monthsFmt)
            ax.autoscale_view()
    fig.autofmt_xdate()

def plot_players_performance(TLGProb_NBA):
    plt.figure(len(TLGProb_NBA.key_to_player_input_id)*2)
    lines = []
    dates = []
    for y, m, d in TLGProb_NBA.game_dict['date']:
        date = y*10000+m*100+d
        if(date not in dates):
            dates.append(date)
    for player in TLGProb_NBA.all_player:
        line_x, line_y = [], []
        for i, date in enumerate(TLGProb_NBA.player_to_date[player]):
            date_ind = bisect_left(dates, date)
            line_x.append(date_ind)
            line_y.append(TLGProb_NBA.player_to_attributes[player]["+/-"][i])
        plt.plot(line_x, line_y)
    plt.title("Time Series of Players' Performances")
    plt.xlabel("Date Index")
    plt.ylabel("Player Performance")

def plot_players_ability(TLGProb_NBA):
    plt.figure(len(TLGProb_NBA.key_to_player_input_id)*2+1)
    lines = []
    dates = []
    for y, m, d in TLGProb_NBA.game_dict['date']:
        date = y*10000+m*100+d
        if(date not in dates):
            dates.append(date)
    for player in TLGProb_NBA.all_player:
        line_x, line_y = [], []
        for i, date in enumerate(TLGProb_NBA.player_to_date[player]):
            date_ind = bisect_left(dates, date)
            line_x.append(date_ind)
            line_y.append(TLGProb_NBA.player_to_attributes[player]["smoothed_plus_minus"][i])
        plt.plot(line_x, line_y)
    plt.title("Time Series of Players' Abilities")
    plt.xlabel("Date Index")
    plt.ylabel("Player Ability")

TLGProb_NBA = TLGProb()
TLGProb_NBA.load_data()
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8), dpi=200)
player = "LeBron James"
feature = "3p"
plot_unsmoothed_feature(TLGProb_NBA, fig, ax[0], "b--", feature, player)
plot_smoothed_feature(TLGProb_NBA, fig, ax[0], "r-", feature, player)
ax[0].legend(loc=9, prop={'size':15})
ax[0].set_ylabel(feature.upper(), fontsize=18)
ax[0].set_title(player+" in NBA 2014/2015 Season")
feature = "fg"
plot_unsmoothed_feature(TLGProb_NBA, fig, ax[1], "b--", feature, player)
plot_smoothed_feature(TLGProb_NBA, fig, ax[1], "r-", feature, player)
ax[1].legend(loc=9, prop={'size':15})
ax[1].set_ylabel(feature.upper(), fontsize=18)
plt.tight_layout()
fig.savefig('../lebron_james_3p_fg.png')