#### Exploring odds data
import pandas as pd
import random
import pickle
import numpy as np


base_name = "data/odds_data/{}.xlsx"

def load_odds_data(filepath, season):

	year_1 = season.split("-")[0]
	year_2 = season.split("-")[1]

	data = pd.read_excel(filepath)

	new_col = ["Date", "Home", "Away", "OU", "Spread", "OU_2H", "Spread_2H", "ML_home", "ML_away", "Points", "Win Margin", "2H Points", "2H Win Margin"]
	new_data = []

	for i in range(0, data.shape[0], 2):
		row1 = data.iloc[[i]]
		row2 = data.iloc[[i + 1]]

		away_team = row1["Team"]
		home_team = row2["Team"]

		date = "-" + str(row1["Date"][i])[:2] + "-" + str(row1["Date"][i])[2:]
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

if __name__ == '__main__':
	season = "2008-2009"
	file = base_name.format(season)
	load_odds_data(file, season)
