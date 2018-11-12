import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np


def compute_averages(file_path, n_games):
	'''
	Computes the last n_games game average for each team.
	@params: file_path (str): csv path where the data is being store
			 n_games (int): number of games we want to look back at
	returns: a new dataframe, this time with each average added to the corresponding game row
	'''

	games = pd.read_csv(file_path)

	def make_averages(n, team_df):
		'''
		makes the average for games in |team_df| over the previous |n_games| (for game n)
		'''
		temp = team_df[int(team_df.index[-1]) < n && int(team_df.index[-1]) >= n - n_games]
		return temp.mean(axis=1)

	average_list = []
	teams = list(pd.unqiue(games.home_name))
	for team in teams:
		team_games = games[team in games.index]
		for i in range(team_games.shape[0]):
			new_list = [team + str(i)]
			new_list.extend(list(make_averages(i, team_df)))
			average_list.append(new_list)

	c = list(my_dataframe.columns.values)
	cols = []
	for col in col:
		cols.append("average_" + col)

	##### finish making the dataframe, make columns correctly, merging the dataframe and returning it


if __name__ == "__main__":

	compute_averages()
