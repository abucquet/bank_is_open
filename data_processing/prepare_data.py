import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
import time
from datetime import date


season_names = {'Golden State Warriors':'GSW',
                'Los Angeles Lakers': 'LAL',
                'San Antonio Spurs': 'SAS',
                'Cleveland Cavaliers': 'CLE',
                'Denver Nuggets': 'DEN',
                'Indiana Pacers': 'IND',
                'Memphis Grizzlies': 'MEM',
                'New Jersey Nets': 'BRK',
                'Brooklyn Nets': 'BRK',
                'New Orleans Hornets': 'NOP',
                'New Orleans Pelicans': 'NOP',
                'Orlando Magic': 'ORL',
                'Toronto Raptors': 'TOR',
                'Miami Heat': 'MIA',
                'Seattle SuperSonics': 'OKC',
                'Utah Jazz': 'UTA',
                'Atlanta Hawks': 'ATL',
                'Boston Celtics': 'BOS',
                'Charlotte Bobcats': 'CHA',
                'Charlotte Hornets': 'CHA',
                'Chicago Bulls': 'CHI',
                'Los Angeles Clippers': 'LAC',
                'Minnesota Timberwolves': 'MIN',
                'Phoenix Suns': 'PHO',
                'Dallas Mavericks': 'DAL',
                'Houston Rockets': 'HOU',
                'Milwaukee Bucks': 'MIL',
                'Philadelphia 76ers': 'PHI',
                'Washington Wizards': 'WAS',
                'Detroit Pistons': 'DET',
                'New York Knicks': 'NYK',
                'Sacramento Kings': 'SAC',
                'Portland Trail Blazers': 'POR',
                'Oklahoma City Thunder': 'OKC'
        }

to_remove = ['team_eFG%', 'team_2P', 'team_ORBr', 'team_2PAr', 'team_3PA',
       'team_2P%', 'team_TOV%', 'team_FT', 'team_STL', 'team_FIC',
       'team_AST/TOV', 'team_TRB', 'team_PTS', 'team_FGA',
       'team_DRB', 'team_AST', 'team_TOV', 'team_FT%', 'team_+/-',
       'team_AST%', 'team_2PA', 'team_TRB%', 'team_3P', 'team_FTA',
       'team_FG%', 'team_3PAr', 'team_DRB%', 'team_USG%', 'team_PF',
       'team_TSA', 'team_FT/FGA', 'opponent_ORB', 'opponent_FIC',
       'opponent_2PA', 'opponent_DRtg', 'opponent_eFG%', 'opponent_2P%',
       'opponent_USG%', 'opponent_2P', 'opponent_ORBr', 'opponent_2PAr',
       'opponent_BLK%', 'opponent_HOB', 'team_BLK', 'opponent_AST/TOV',
       'opponent_FG%', 'opponent_TOV%', 'opponent_FT', 'opponent_+/-',
       'opponent_AST%', 'opponent_TOV', 'opponent_FGA', 'opponent_PTS',
       'opponent_STL', 'opponent_TSA', 'opponent_PF', 'opponent_TRB%',
       'opponent_3PAr', 'opponent_3P', 'team_3P%', 'opponent_TRB',
       'opponent_AST', 'opponent_DRB', 'opponent_FT%', 'opponent_FTA',
       'opponent_DRB%']


def compute_day_diff(d1, d2):
    # d1 = date(int(str(d1)[:4]), int(str(d1)[4:6]), int(str(d1)[6:8]))
    # d2 = date(int(str(d2)[:4]), int(str(d2)[4:6]), int(str(d2)[6:8]))
    
    return (d2 - d1).days

def getData(year):
    game_data_path = "data/final_game_data/"
    game_filepath = str(year) + "-" + str(year + 1) + "_games_final.csv"
    season = pd.read_csv(game_data_path + game_filepath)

    #season = season.drop(to_remove, axis = 1)
    
    odds_data_path = "data/odds_data_processed/"
    odds_filepath = str(year) + "-" + str(year + 1) + ".csv"
    odds = pd.read_csv(odds_data_path + odds_filepath)
    odds = odds.drop(['Unnamed: 0'], axis = 1)
    
    return season, odds

def cleanNames(season, odds):
    odds_names = {}
    for name in list(pd.unique(odds.Home)):
        found = False
        for s_name in season_names:
            if name in s_name:
                found = True
                odds_names[name] = season_names[s_name]
    odds_names["LA Lakers"] = "LAL"
    odds_names["LA Clippers"] = "LAC"
    odds_names["Okla City"] = "OKC"
    odds_names["Oklahoma City"] = "OKC"
    
    odds["Home"] = odds["Home"].apply(lambda x: odds_names[x])
    odds["Away"] = odds["Away"].apply(lambda x: odds_names[x])

    season["team"] = season["team"].apply(lambda x: season_names[x])
    season["opponent"] = season["opponent"].apply(lambda x: season_names[x])
    
    return season, odds

def make_index(row, col1, col2, col3):
    return str(row[col1]) + str(row[col2]) + str(row[col3])

def find_category(row, odds):
    ref = row["Index"]
    if row["home"] == 0:
        ref = ref[:-6] + ref[-3:] + ref[-6:-3]
    odds_row = odds.loc[odds["Index"] == ref]
    try:
        return list(odds_row["Points"])[0]
    except:
        return 0

def makeIndices(season, odds):
    season["date"] = season["date"].apply(lambda x: str(x)[:-1])
    season["Index"] = season.apply(lambda x: make_index(x, "date", "team", "opponent"), axis=1)
    
    odds["Date"] = odds["Date"].apply(lambda x: "".join(x.split("-")))
    odds["Index"] = odds.apply(lambda x: make_index(x, "Date", "Home", "Away"), axis=1)
    
    season["Outcome"] = season.apply(lambda x: find_category(x, odds), axis = 1) ##### CHANGE THIS TO DEAL WITH OTHER INDICES
    
    in_data = season.set_index("Index")
    in_data = in_data.drop(["index"], axis = 1)
    in_data = in_data.sort_index()
    
    return season, odds, in_data

def compute_season_averages(in_data, dates):
    season_averages = {}

    for date in dates:
        # get all past games
        past_games = in_data[in_data.date < date]
        # means
        season_averages[date] = past_games.groupby('team').mean()

    return season_averages

def get_past_n(in_data, dates, n):
    ## build a list of games for every team
    past_n = {}
    home_only = in_data[in_data.home == 1]

    for date in dates:
        team_map = {}
        past_games = in_data[in_data.date < date]
        for team in pd.unique(home_only.team):
            #get the past games for team
            past_team = past_games[past_games.team == team].tail(3)
            past_team["time_ago"] = past_team["date"].apply(lambda x: compute_day_diff(x, date))

            team_map[team] = past_team
        past_n[date] = team_map 

    return past_n

def one_hot_encode_teams():
    ## one-hot encode team names
    teams = season_names.values()
    encoding = {}
    index = 0
    for team in teams:
        if team not in encoding:
            encoding[team] = index
            index += 1

    empty_list = [0 for j in range(index + 1)]
    encoded = {}
    for team in teams:
        if team in encoded: continue

        copy = empty_list[:]

        i = encoding[team]
        copy[i] = 1
        encoded[team] = copy

    return encoded

def make_date(d1):
    return date(int(str(d1)[:4]), int(str(d1)[4:6]), int(str(d1)[6:8]))

def createFinalData(in_data):

    with open("data/distance_map.pkl", "rb") as f:
        distance_map = pickle.load(f)
    
    in_data["date"] = in_data["date"].apply(make_date)

    dates = pd.unique(in_data.date)

    n = 3
    home_only = in_data[in_data.home == 1]

    season_averages = compute_season_averages(in_data, dates)

    past_n = get_past_n(in_data, dates, n)

    encoded = one_hot_encode_teams()

    X = []
    y = []

    for i, row in home_only.iterrows():

        home_team = row["team"]
        away_team = row["opponent"]

        date = row["date"]

        past_n_home = past_n[date][home_team]
        past_n_away = past_n[date][away_team]

        avgs = season_averages[date]

        if past_n_home.shape[0] < n or past_n_away.shape[0] < n: continue

        ################ HOME TEAM PAST GAMES
        data_home = []
        i = 0
        for j, row_2 in past_n_home.iterrows():
            cur_data = []

            team = row_2["team"]
            opponent = row_2["opponent"]

            cur_data.extend(encoded[team])
            cur_data.extend(encoded[opponent])
            cur_data.extend(row_2.drop(["team", "opponent", "date"]).values)

            # distance to next game
            if i != n - 1:
                next_game = past_n_home.iloc[i + 1]
            else:
                next_game = row

            city1 = team if row_2["home"] == 1 else row_2["opponent"]
            city2 = team if next_game["home"] == 1 else next_game["opponent"]
            dist_to_travel = distance_map[(city1, city2)]
            cur_data.append(dist_to_travel)


            opp_stats = avgs.loc[opponent].values

            cur_data.extend(opp_stats)

            data_home.append(cur_data)

            i += 1

        ################ AWAY TEAM PAST GAMES
        data_away = []
        i = 0
        for j, row_2 in past_n_away.iterrows():
            cur_data = []

            team = row_2["team"]
            opponent = row_2["opponent"]

            cur_data.extend(encoded[team])
            cur_data.extend(encoded[opponent])
            cur_data.extend(row_2.drop(["team", "opponent", "date"]).values)

            # distance to next game
            if i != n - 1:
                next_game = past_n_home.iloc[i + 1]
            else:
                next_game = row

            city1 = team if row_2["home"] == 1 else row_2["opponent"]
            city2 = team if next_game["home"] == 1 else next_game["opponent"]
            dist_to_travel = distance_map[(city1, city2)]
            cur_data.append(dist_to_travel)

            opp_stats = avgs.loc[opponent].values

            cur_data.extend(opp_stats)

            data_away.append(cur_data)

            i += 1

        ################ MERGE THE TWO
        data = []
        for i in range(len(data_home)):
            cur_data = data_home[i]
            cur_data.extend(data_away[i])
            data.append(cur_data)

        X.append(data)
        y.append(row["Outcome"])

    return np.array(X), np.array(y)


if __name__ == '__main__':
    for year in tqdm(range(2007, 2018)):
        if year == 2011: continue
        season, odds = getData(year)
        season, odds = cleanNames(season, odds)
        season, odds, in_data = makeIndices(season, odds)

        with open("temp.pkl", "wb") as f:
            pickle.dump(in_data, f)

        X, y = createFinalData(in_data)

        with open("data/neural_net_data/" + str(year) + "-" + str(year + 1) + ".pkl", 'wb') as f:
            pickle.dump((X, y), f)