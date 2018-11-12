#### Reading the json file data


# What we need to know about a given game between two teams:
### Per game info:


import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np

def compile_season(season, directory="basketball_reference-master/matches/united_states/nba/"):
	'''
	Compiles the data from a season by making two daraframes: one for every team-game, and one for every player-game
	@params: season (str): the season we want to be looking at
			 directory (str): the path of the folder containing all seasons
	returns: two dataframes
	'''
	files = os.listdir(directory + season)

	team_game_index = defaultdict(int)

	season_data = pd.DataFrame() ### need to specify columns

	for file in files:
		if ".json" not in file: continue

		with open(directory + season + "/" + file) as json_data:
			    game_data = json.load(json_data)
			    json_data.close()

		team_game_index[game_data["home"]["name"]] += 1
		team_game_index[game_data["away"]["name"]] += 1
		
		#################### GAME TABLE
		## initialize the data entries
		stats = {}

		## location ??
		#stats["Location"] = 

		## date
		stats["date"] = game_data["code"][:-3]

		## team names
		stats["home_name"] = game_data["home"]["name"]
		stats["away_name"] = game_data["away"]["name"]

		for name, value in game_data["home"]["totals"].iteritems():
			stats["home_" + name] = value

		for name, value in game_data["away"]["totals"].iteritems():
			stats["away_" + name] = value

		## adding the stats to the df
		# for home team
		stats["index"] = game_data["home"]["name"] + str(team_game_index[game_data["home"]["name"]])
		if season_data.shape[0] == 0:
			season_data = pd.DataFrame(columns=stats.keys())
		season_data = season_data.append(stats, ignore_index=True)

		# for away team
		stats["index"] = game_data["away"]["name"] + str(team_game_index[game_data["away"]["name"]])
		season_data = season_data.append(stats, ignore_index=True)

	print(team_game_index)

	return season_data

if __name__ == "__main__":

	df = compile_season("2007-2008")
	print(df.shape)
