#### Reading the json file data


# What we need to know about a given game between two teams:
### Per game info:


import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm

PROCESSED_PATH = "./processed_game_data/"
PROCESSED_PATH_FORMAT = "./processed_game_data/{}.csv"

def compile_season(season, directory="basketball_reference-master/matches/united_states/nba/"):
	'''
	Compiles the data from a season by making two daraframes: one for every team-game, and one for every player-game
	@params: season (str): the season we want to be looking at
			 directory (str): the path of the folder containing all seasons
	returns: two dataframes
	'''
	files = os.listdir(directory + season)
	print("Found {0} files for season {1}".format(len(files), season))

	team_game_index = defaultdict(int)

	season_data = pd.DataFrame() ### need to specify columns
	player_date_data = pd.DataFrame()

	for file in tqdm(files):
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

		for name, value in game_data["home"]["totals"].items():
			stats["home_" + name] = value

		for name, value in game_data["away"]["totals"].items():
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

	# print(team_game_index)
		continue
		###################### PLAYER TABLE
		players = defaultdict(list)
		## One row per player per game
		for team in ["home","away"]:	# "home" and "away". Should loop over keys
			for entry in game_data[team]["players"]:
				# Get team for player (in case they switch teams)
				# Loop over players and fill table
				players["date"].append(game_data["code"][:-3])
				players["team"].append(game_data[team]["name"])
				for name, value in game_data[team]["players"][entry].items():
					players[name].append(value)

				#player_date_data = player_date_data.append(player, ignore_index=True)

		#player_date_data = pd.DataFrame(columns=players.keys())
		#player_date_data = player_date_data.append(players, ignore_index=True)

		player_date_data = pd.DataFrame(players)


	## SOME SORT OF ERROR CHECKING TO MAKE SURE WE HAVE ALL THE DATA FOR A SEASON:
	# Make sure we have all the regular-season games
	# Make sure that player data is complete
	# Simple sanity check on the numbers making sense

	return season_data, player_date_data

def get_seasons(seasons):
    """
    gets seasons involved in range
    """
    if '-to-' in seasons:
        from_, to = map(int, seasons.split('-to-'))
        years = list(range(from_, to+1))
        seasons = []
        for i in range(0, len(years)-1):
            season = "-".join(map(str, [years[i], years[i+1]]))
            seasons.append(season)
    return seasons


if __name__ == "__main__":

	if not os.path.exists(PROCESSED_PATH):
		os.mkdir(PROCESSED_PATH)

	seasons = "2007-to-2011"
	seasons = get_seasons(seasons)
	print(seasons)

	for season in seasons:
		df, player_df = compile_season(season)
		print(df.shape)
		df.to_csv(PROCESSED_PATH_FORMAT.format(season + "_games"))
		player_df.to_csv(PROCESSED_PATH_FORMAT.format(season + "_players"))
