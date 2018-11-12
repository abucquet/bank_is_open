#### Reading the json file data


# What we need to know about a given game between two teams:
### Per game info:


import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np

directory = "bank_is_open/basketball_reference-master/matches/united_states/nba/"
season = "2007-2008"
files = os.listdir(directory + season)

season_avg_teams = defaultdict(list)
season_avg_players = defaultdict(list)
last_ten_avg_teams = defaultdict(list)
last_ten_teams = defaultdict(list)

team_game_index = {}

season_data = pd.DataFrame() ### need to specify columns

for file in files:
	if ".json" not in file: continue

	with open(file) as json_data:
		    game_data = json.load(json_data)
		    json_data.close()

	
	#################### GAME TABLE
	## initialize the data entries
	stats = {}

	## location ??
	#stats["Location"] = 

	## index: team1-team2
	stats["index"] = game_data["home"]["name"]+game_data["away"]["name"]

	## date
	stats["date"] = game_data["code"][:-3]

	## team names
	stats["home_name"] = game_data["home"]["name"]
	stats["away_name"] = game_data["away"]["name"]

	for name, value in game_data["home"]["totals"].iteritems():
		stats["home_" + name] = value

	for name, value in game_data["away"]["totals"].iteritems():
		stats["away_" + name] = value


	###################### PLAYER TABLE
	## One row per player per game
	players = {}
	for team in game_data:	# "home" and "away". Should loop over keys
		for entry in game_data[team]:
			# Get team for player (in case they switch teams)
			# Loop over players and fill table



	## SOME SORT OF ERROR CHECKING TO MAKE SURE WE HAVE ALL THE DATA FOR A SEASON:
	# Make sure we have all the regular-season games
	# Make sure that player data is complete
	# Simple sanity check on the numbers making sense



print(game_data["code"])
print(game_data["away"]["name"])
#print(game_data["away"]["totals"])
print(game_data["away"]["scores"])
print(game_data["home"]["name"])
#print(game_data["home"]["totals"])
print(game_data["home"]["scores"])

