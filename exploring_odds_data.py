#### Exploring odds data
import pandas as pd
import random
import pickle
import numpy as np
from tqdm import tqdm
from scipy import stats
import os


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

ODDS_PATH = "./data/odds_data/"
ODDS_PATH_PROCESSED = "./data/odds_data_processed/{}.csv"


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
		if type(OU) not in [np.float64, np.int64] or type(spread) not in [np.float64, np.int64]:
			continue

		OU_Spread_2H = np.sort([row1["2H"][i], row2["2H"][i+1]])
		OU_2H = OU_Spread_2H[1]
		spread_2H = OU_Spread_2H[0]

		ML_h = row2["ML"][i+1]
		ML_a = row1["ML"][i]
		
		# if the ML of the home is less than tat of the away, the home team is favorite so we readjust the spread
		if ML_h > ML_a:
			if type(spread) == int:
				spread = -spread
				spread_2H = -spread_2H


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

	data = pd.DataFrame(new_rows, columns=cols)

	
	# final cleaning of the data
	stripped_cols = [c for c in cols if "Team" not in c]

	data = data[stripped_cols]

	data = data.dropna(axis = 1)

	data = data[(data[data.columns] != "--").all(axis=1)]

	return data



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

def chisquare(season_data):
	'''
	Runs a chi-square test of correlation between the betting indicators in the data
	Returns: nothing
	'''
	counts_OU_spread = np.zeros((2, 2))
	counts_OU_ML = np.zeros((2, 2))

	OU_list = []
	ML_list = []
	spread_list = []

	errors = 0

	for i, row in season_data.iterrows():
		OU = 0
		ML = 0
		spread = 0
		try:
			if season_data["OU"][i] > season_data["Points"][i]: 
				OU = 1

			if season_data["Spread"][i] > season_data["Win Margin"][i]: 
				spread = 1

			direction_ML = 1 if season_data["ML_home"][i] < season_data["ML_away"][i]  else -1

			if direction_ML*season_data["Win Margin"][i] > 0: 
				ML = 1

			counts_OU_spread[OU][spread] += 1
			counts_OU_ML[OU][ML] += 1


			OU_list.append(OU)
			ML_list.append(ML)
			spread_list.append(spread)

		except:
			errors += 1

	print("----------------------")
	print("Chi-squared test results")


	print("Over-Under vs. Spread")
	#print(counts_OU_spread)
	test_stat, degrees_freedom, p = run_chisquare(counts_OU_spread)
	print("The test statistic is {} and has {} degrees of freedom. This gives a p-value of {}".format(test_stat, degrees_freedom, p))

	print("----")

	print("Over-Under vs. Money-Line")
	#print(counts_OU_ML)
	test_stat, degrees_freedom, p = run_chisquare(counts_OU_ML)
	print("The test statistic is {} and has {} degrees of freedom. This gives a p-value of {}".format(test_stat, degrees_freedom, p))
	
	print("----------------------")
	print("")

	# # run pearson's r^2
	# print("########### Pearson's R")
	# print("Over-Under vs. Money Line")
	# print(stats.pearsonr(OU_list, ML_list))
	# print("Over-Under vs. Spread")
	# print(stats.pearsonr(OU_list, spread_list))
	# print("Spread vs. Money Line")
	# print(stats.pearsonr(spread_list, ML_list))
	# print("")

	#print("In the process, we have had {} errors.".format(errors))
	#print("")

def run_chisquare(count_data):
	'''
	Runs a chi-squared test on the categorical data |count_data|, a 2D numpy array
	'''

	degrees_freedom = (count_data.shape[0] - 1) * (count_data.shape[1] - 1)

	row_counts = np.sum(count_data, axis=0)
	col_counts = np.sum(count_data, axis=1)
	total_sum = np.sum(count_data)

	expected_counts = np.outer(row_counts, col_counts)/total_sum

	test_stat = count_data - expected_counts
	test_stat = test_stat * test_stat
	test_stat = test_stat / expected_counts
	test_stat = np.sum(test_stat)

	return test_stat, degrees_freedom, (1 - stats.chi2.cdf(test_stat, degrees_freedom))

if __name__ == '__main__':
	files = os.listdir(ODDS_PATH)
	for file in files:

		print("")
		print(file)
		print("")

		if "odds" not in file: continue
		path = ODDS_PATH + file
		season = int(file.split(" ")[2].split("-")[0])

		season_str = str(season) + "-" + str(season+1)

		save_to = ODDS_PATH_PROCESSED.format(season_str)

		odds_data = load_odds_data(path, season_str)

		odds_data.to_csv(ODDS_PATH_PROCESSED)

		chisquare(odds_data)

	#season_data = merge_with_features(odds_data)
	
	#print(season_data.shape)

	# with open(save_to.format(season), "wb") as f:
	# 	pickle.dump(season_data, f)
