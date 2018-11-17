#### Exploring odds data
import pandas as pd
import random
import pickle
import numpy as np
from tqdm import tqdm


base_name = "data/odds_data/{}.xlsx"
games_folder = "data/matchups/"
save_to = "data/aggregated_{}.pkl"

team_map = {
			'LALakers': 'LA Lakers', 
			'LAClippers': 'LA Clippers',
			'NewYork' : 'New York',
			'OklahomaCity': 'Okla City',
			'SanAntonio' : 'San Antonio',
			'GoldenState': 'Golden State',
			'NewOrleans': 'New Orleans', 
			'NewJersey': 'Brooklyn'
			}


def load_odds_data(filepath, season):
	'''
	Loads the data from filepath and stores it in a pandas DataFrame that contains one row per game.
	Returns: a pandas DataFrame
	'''

	year_1 = season.split("-")[0]
	year_2 = season.split("-")[1]

	data = pd.read_excel(filepath)

	new_col = ["Date", "Home", "Away", "OU", "Spread", "OU_2H", "Spread_2H", "ML_home", "ML_away", "Points", "Win Margin", "2H Points", "2H Win Margin"]
	new_data = []

	for i in range(0, data.shape[0], 2):
		row1 = data.iloc[[i]]
		row2 = data.iloc[[i + 1]]

		away_team = row1["Team"][i]
		home_team = row2["Team"][i + 1]

		if away_team in team_map:
			away_team = team_map[away_team]
		if home_team in team_map:
			home_team = team_map[home_team]

		
		if len(str(row1["Date"][i])) == 4:
			day = str(row1["Date"][i])[2:]
			month = str(row1["Date"][i])[:2]
		else:
			day = str(row1["Date"][i])[1:]
			month = "0" + str(row1["Date"][i])[:1]

		date = "-" + month + "-" + day
		if int(str(row1["Date"][i])[:2]) > 7:
			date = year_1 + date
		else:
			date = year_2 + date


		OU_Spread = np.sort([row1["Open"][i], row2["Open"][i+1]])
		OU = OU_Spread[1]
		spread = OU_Spread[0]

		OU_Spread_2H = np.sort([row1["2H"][i], row2["2H"][i+1]])
		OU_2H = OU_Spread_2H[1]
		spread_2H = OU_Spread_2H[0]

		ML_h = row2["ML"][i+1]
		ML_a = row1["ML"][i]

		win_margin = row2["Final"][i+1] - row1["Final"][i]
		final_points = row2["Final"][i+1] + row1["Final"][i]

		h2_win_margin = row2["3rd"][i+1] - row1["3rd"][i] + row2["4th"][i+1] - row1["4th"][i]
		h2_final_points = row2["3rd"][i+1] + row1["3rd"][i] + row2["4th"][i+1] + row1["4th"][i]

		new_row = [date, home_team, away_team, OU, spread, OU_2H, spread_2H, ML_h, ML_a, final_points, win_margin, h2_final_points, h2_win_margin]
		new_data.append(new_row)

	updated_data = pd.DataFrame(new_data, columns=new_col)
	
	return updated_data

def merge_with_features(odds_data):
	'''
	Merges each row in odds data (pandas DataFrame) with the appropriate game statistics
	Returns an updated DataFrame
	'''

	old_cols = list(odds_data.columns)
	new_cols = []
	new_rows = []

	for i, row in tqdm(odds_data.iterrows()):
		date = row["Date"]
		home_team = row["Home"]
		away_team = row["Away"]

		n_c, new_row = get_data_from_pickle(date, home_team, away_team)
		if len(n_c) > len(new_cols):
			new_cols = n_c
		
		new_row = list(row) + new_row
		new_rows.append(new_row)

	cols = old_cols + new_cols

	return pd.DataFrame(new_rows, columns=cols)



def get_data_from_pickle(date, home_team, away_team):
	'''
	Looks in the file for the specific date and retrieves information for both home and away teams.
	Returns a tuple with (column names, data)
	'''

	with open(games_folder + date + ".pkl", "rb") as f:
		game_day = pickle.load(f)
		f.close()

	cols = game_day.columns

	home_cols = ["home_" + col for col in cols]
	away_cols = ["away_" + col for col in cols]

	home = game_day.loc[game_day["Team"].isin([home_team])]
	away = game_day.loc[game_day["Team"].isin([away_team])]

	data = list(home.iloc[0]) + list(away.iloc[0])

	return (home_cols + away_cols, data)


if __name__ == '__main__':
	season = "2008-2009"
	file = base_name.format(season)
	odds_data = load_odds_data(file, season)
	print("Loaded odds data.")

	season_data = merge_with_features(odds_data)
	
	with open(save_to.format(season), "wb") as f:
		pickle.dump(season_data, f)
