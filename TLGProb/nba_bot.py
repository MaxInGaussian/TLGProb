################################################################################
#  TLGProb: Two-Layer Gaussian Process Regression Model For
#           Winning Probability Calculation of Two-Team Sports
#  Github: https://github.com/MaxInGaussian/TLGProb
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import os
import csv
from sklearn.externals.six.moves import xrange
from .bot import Bot

class NBA_Bot(Bot):

    all_player, player_dict = set(), {}

    def __init__(self, database_path="NBA_database/"):
        super(basketball_bot, self).__init__()
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        self.database_path = os.path.join(fileDir, database_path)

    def search_all_basketball_player(self):
        c = ord('a')
        while c <= ord('z'):
            data_dict = self.search_basketball_player_by_name_char(chr(c))
            print("Found", len(data_dict["url"]), "players with", chr(c))
            self.save_basketball_data(data_dict)
            c += 1

    def get_basketball_game_log_by_year(self, year, update=True):
        self.load_basketball_all_player_data()
        dir_path = self.database_path+str(year-1)[2:]+"-"+str(year)[2:]+"/"
        if(not os.path.exists(dir_path)):
            os.makedirs(dir_path)
        headers = ["date", "team", "opp", "age", "gs", "mp", "fg",
                   "fga", "fg%", "3p", "3pa", "3p%", "ft", "fta",
                   "ft%", "orb", "drb", "trb", "ast", "stl",
                   "blk", "tov", "pf", "pts", "gmsc", "+/-"]
        for i in xrange(len(self.all_player)):
            if(self.player_dict["from"][i] <=
               year <= self.player_dict["to"][i]):
                rename = self.player_dict["name"][i].replace(" ", "_")
                player_log_csv = dir_path + rename + ".csv"
                first_save = not os.path.isfile(player_log_csv) or not update
                load_date_log = []
                if(not first_save):
                    with open(player_log_csv, 'rt') as f:
                        reader = csv.reader(f)
                        headers = next(reader)
                        for row in reader:
                            for h, v in zip(headers, row):
                                if(h.lower() == "date"):
                                    load_date_log.append(v)
                player_url = self.player_dict["url"][i]
                game_log_url = player_url[:-5] + "/gamelog/" + str(year)
                data_dict = self.get_basketball_game_log_by_url(
                    game_log_url, load_date_log, update)
                with open(player_log_csv, 'wt' if first_save else 'at') as f:
                    csv_writer = csv.DictWriter(f, fieldnames=headers)
                    if(first_save):
                        csv_writer.writeheader()
                    for j, date in enumerate(data_dict["date"]):
                        if(date not in load_date_log):
                            csv_dict = dict.fromkeys(headers)
                            for k in headers:
                                csv_dict[k] = data_dict[k][j]
                            csv_writer.writerow(csv_dict)

    def get_all_games_by_year(self, year, update=True):
        data_dict = {"date": [], "team1": [], "team1pts": [],
                     "team2": [], "team2pts": []}
        key_to_i = {"date": 0, "team1": 3, "team1pts": 4,
                    "team2": 5, "team2pts": 6}
        url = "http://www.basketball-reference.com/leagues/NBA_" +\
            str(year) + "_games.html"
        br_id = self._get_any_idle_br_id()
        self._new_command(br_id, (self.GOTO, url))
        rows = self._new_command(br_id, (self.ID, "games",
                                         self.TAG, "tr"))[1]
        for row in rows:
            cols = self._find_elements(row, self.TAG, "td")
            if(len(cols) < 9):
                continue
            for key, i in key_to_i.items():
                s = str(cols[i].text.replace(" ", ""))
                if(key == "team1pts" or key == "team2pts"):
                    if(len(cols[i].text) == 0):
                        data_dict[key].append(-1)
                    else:
                        data_dict[key].append(float(s))
                elif(key == "date"):
                    from datetime import datetime
                    date = datetime.strptime(str(s), '%a,%b%d,%Y')
                    data_dict[key].append(date.strftime("%Y-%m-%d"))
                else:
                    name_a = self._find_elements(cols[i], self.TAG, "a")
                    if(len(name_a) != 1):
                        print(s + "Error!")
                    link = str(name_a[0].get_attribute("href"))
                    data_dict[key].append(link.split("/")[-2])
        dir_path = self.database_path
        all_game_csv = dir_path + "all_games.csv"
        headers = ["date", "team1", "team1pts", "team2", "team2pts"]
        first_save = not os.path.isfile(all_game_csv) or not update
        ori_rows, load_date_log = [], set()
        if(not first_save):
            with open(all_game_csv, 'rt') as f:
                reader = csv.reader(f)
                headers = next(reader)
                for row in reader:
                    ori_rows.append(row)
                    for h, v in zip(headers, row):
                        if(h.lower() == "date"):
                            load_date_log.add(v)
        with open(all_game_csv, 'wt') as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            csv_writer.writeheader()
            if(not first_save):
                for row in ori_rows:
                    csv_dict = dict.fromkeys(headers)
                    csv_dict["date"] = row[0]
                    csv_dict["team1"] = row[1]
                    csv_dict["team1pts"] = row[2]
                    csv_dict["team2"] = row[3]
                    csv_dict["team2pts"] = row[4]
                    if(float(row[2]) > 0):
                        csv_writer.writerow(csv_dict)
                        continue
                    if(float(row[2]) < 0):
                        for i, date in enumerate(data_dict["date"]):
                            if(row[0] == date and row[1] == data_dict["team1"][i]):
                                csv_dict["team1pts"] = data_dict["team1pts"][i]
                                csv_dict["team2pts"] = data_dict["team2pts"][i]
                        csv_writer.writerow(csv_dict)
            for i, date in enumerate(data_dict["date"]):
                if(date not in load_date_log):
                    csv_dict = dict.fromkeys(headers)
                    csv_dict["date"] = data_dict["date"][i]
                    csv_dict["team1"] = data_dict["team1"][i]
                    csv_dict["team1pts"] = data_dict["team1pts"][i]
                    csv_dict["team2"] = data_dict["team2"][i]
                    csv_dict["team2pts"] = data_dict["team2pts"][i]
                    csv_writer.writerow(csv_dict)

    def get_basketball_game_log_by_url(self, game_log_url, log, update):
        data_dict = {"date": [], "team": [], "opp": [], "age": [], "gs": [],
                     "mp": [], "fg": [], "fga": [], "fg%": [], "3p": [],
                     "3pa": [], "3p%": [], "ft": [], "fta": [], "ft%": [],
                     "orb": [], "drb": [], "trb": [], "ast": [], "stl": [],
                     "blk": [], "tov": [], "pf": [], "pts": [], "gmsc": [],
                     "+/-": []}
        key_to_i = {"date": 2, "team": 4, "opp": 6, "age": 3, "gs": 8,
                    "mp": 9, "fg": 10, "fga": 11, "fg%": 12, "3p": 13,
                    "3pa": 14, "3p%": 15, "ft": 16, "fta": 17, "ft%": 18,
                    "orb": 19, "drb": 20, "trb": 21, "ast": 22, "stl": 23,
                    "blk": 24, "tov": 25, "pf": 26, "pts": 27, "gmsc": 28,
                    "+/-": 29}
        br_id = self._get_any_idle_br_id()
        self._new_command(br_id, (self.GOTO, game_log_url))
        rows = self._new_command(br_id, (self.ID, "pgl_basic",
                                         self.TAG, "tr"))[1]
        for row in rows:
            cols = self._find_elements(row, self.TAG, "td")
            if(len(cols) < 30):
                continue
            if(update and str(
                    cols[key_to_i["date"]].text.replace(" ", "")) in log):
                continue
            for key, i in key_to_i.items():
                if(len(cols[i].text) == 0):
                    data_dict[key].append(0.)
                    continue
                s = str(cols[i].text.replace(" ", ""))
                if(key == "age"):
                    year, day = map(float, s.split("-"))
                    data_dict[key].append(year+day*1./365.)
                elif(key == "mp"):
                    mins, secs = map(float, s.split(":"))
                    data_dict[key].append(mins+secs*1./60.)
                elif(key == "date" or key == "team" or key == "opp"):
                    data_dict[key].append(s)
                else:
                    data_dict[key].append(float(s))
        return data_dict

    def load_basketball_all_player_data(self):
        dir_path = self.database_path
        all_player_csv = dir_path + "all_players.csv"
        if(not os.path.isfile(all_player_csv)):
            self.all_player = set()
            return
        with open(all_player_csv, 'rt') as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.player_dict = {h: [] for h in headers}
            for row in reader:
                for h, v in zip(headers, row):
                    if(h.lower() == "from"):
                        self.player_dict[h].append(int(v))
                    elif(h.lower() == "to"):
                        self.player_dict[h].append(int(v))
                    elif(h.lower() == "height"):
                        self.player_dict[h].append(float(v))
                    elif(h.lower() == "weight"):
                        self.player_dict[h].append(float(v))
                    elif(h.lower() == "name"):
                        self.all_player.add(v)
                        self.player_dict[h].append(str(v))
                    else:
                        self.player_dict[h].append(str(v))

    def save_basketball_all_player_data(self, data_dict):
        self.load_basketball_all_player_data()
        dir_path = self.database_path
        all_player_csv = dir_path + "all_players.csv"
        headers = ["name", "from", "to", "position", "height",
                   "weight", "birthday", "url"]
        first_save = (len(self.all_player) == 0)
        with open(all_player_csv, 'wt' if first_save else 'at') as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            if(first_save):
                csv_writer.writeheader()
            for i, name in enumerate(data_dict["name"]):
                if(name not in self.all_player):
                    self.all_player.add(name)
                    csv_dict = dict.fromkeys(headers)
                    csv_dict["url"] = data_dict["url"][i]
                    csv_dict["name"] = data_dict["name"][i]
                    csv_dict["from"] = data_dict["from"][i]
                    csv_dict["to"] = data_dict["to"][i]
                    csv_dict["position"] = data_dict["position"][i]
                    csv_dict["height"] = data_dict["height"][i]
                    csv_dict["weight"] = data_dict["weight"][i]
                    csv_dict["birthday"] = data_dict["birthday"][i]
                    csv_writer.writerow(csv_dict)

    def get_basketball_player_by_name_char(self, c, update=True):
        if(update):
            self.load_basketball_all_player_data()
        data_dict = {"name": [], "from": [], "to": [], "position": [],
                     "height": [], "weight": [], "birthday": [], "url": []}
        url = "http://www.basketball-reference.com/players/" + c
        br_id = self._get_any_idle_br_id()
        self._new_command(br_id, (self.GOTO, url))
        rows = self._new_command(br_id, (self.ID, "players",
                                         self.TAG, "tr"))[1]
        for row in rows:
            cols = self._find_elements(row, self.TAG, "td")
            if(len(cols) != 8):
                continue
            name_a = self._find_elements(cols[0], self.TAG, "a")
            if(len(name_a) != 1):
                continue
            link = str(name_a[0].get_attribute("href"))
            if(url not in link):
                print(link)
                continue
            if(len(cols[0].text) == 0 or len(cols[1].text) == 0 or
               len(cols[2].text) == 0 or len(cols[3].text) == 0 or
               len(cols[4].text) == 0 or len(cols[5].text) == 0 or
               len(cols[6].text) == 0):
                continue
            name = str(cols[0].text)
            if(update and name in self.all_player):
                continue
            data_dict["url"].append(link)
            data_dict["name"].append(name)
            data_dict["from"].append(int(cols[1].text.replace(' ', '')))
            data_dict["to"].append(int(cols[2].text.replace(' ', '')))
            data_dict["position"].append(str(cols[3].text).replace(' ', ''))
            feet, inch = map(int, str(cols[4].text).replace(
                ' ', '').split('-'))
            data_dict["height"].append(0.3048*feet+0.0254*inch)
            data_dict["weight"].append(int(cols[5].text.replace(' ', '')))
            data_dict["birthday"].append(str(cols[6].text.replace(' ', '')))
        self.save_basketball_all_player_data(data_dict)

    def get_basketball_player_by_year(self, year, update=True):
        if(update):
            self.load_basketball_all_player_data()
        data_dict = {"name": [], "from": [], "to": [], "position": [],
                     "height": [], "weight": [], "birthday": [], "url": []}
        c = ord('a')
        while c <= ord('z'):
            url = "http://www.basketball-reference.com/players/" + chr(c)
            br_id = self._get_any_idle_br_id()
            self._new_command(br_id, (self.GOTO, url))
            rows = self._new_command(br_id, (self.ID, "players",
                                             self.TAG, "tr"))[1]
            for row in rows:
                cols = self._find_elements(row, self.TAG, "td")
                if(len(cols) != 8):
                    continue
                name_a = self._find_elements(cols[0], self.TAG, "a")
                if(len(name_a) != 1):
                    continue
                link = str(name_a[0].get_attribute("href"))
                if(url not in link):
                    print(link)
                    continue
                if(len(cols[0].text) == 0 or len(cols[1].text) == 0 or
                   len(cols[2].text) == 0 or len(cols[3].text) == 0 or
                   len(cols[4].text) == 0 or len(cols[5].text) == 0 or
                   len(cols[6].text) == 0):
                    continue
                from_ = int(cols[1].text.replace(' ', ''))
                to_ = int(cols[2].text.replace(' ', ''))
                if(not(from_ <= year <= to_)):
                    continue
                name = str(cols[0].text)
                if(update and name in self.all_player):
                    continue
                data_dict["url"].append(link)
                data_dict["name"].append(name)
                data_dict["from"].append(from_)
                data_dict["to"].append(to_)
                data_dict["position"].append(
                    str(cols[3].text).replace(' ', ''))
                feet, inch = map(int, str(cols[4].text).replace(
                    ' ', '').split('-'))
                data_dict["height"].append(0.3048*feet+0.0254*inch)
                data_dict["weight"].append(int(cols[5].text.replace(' ', '')))
                data_dict["birthday"].append(
                    str(cols[6].text.replace(' ', '')))
            c += 1
        self.save_basketball_all_player_data(data_dict)
