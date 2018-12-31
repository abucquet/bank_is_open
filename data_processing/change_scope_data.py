import numpy as np
import pandas as pd
import os

game_data_path = "data/processed_game_data/"
new_game_data_path = "data/final_game_data/"
files = os.listdir(game_data_path)

for file in files:
    if "games.csv" not in file: continue
        
    print(file)
    
    season = pd.read_csv(game_data_path + file)
    if(season.shape[0] == 0): continue
    season = season.drop(["Unnamed: 0"], axis = 1)
    season = season.drop_duplicates(subset=["date", "home_name", "away_name"])

    ######## Compute new columns
    cols = season.columns

    away_cols = []
    home_cols = []
    neutral_cols = []
    stripped_cols = []

    for col in cols:
        if "away" in col and "name" not in col:
            away_cols.append(col)
            stripped_cols.append(col.split("_")[1])
        elif "home" in col and "name" not in col:
            home_cols.append(col)
        elif "name" not in col:
            neutral_cols.append(col)

    new_cols = ["team", "opponent"]
    new_cols.extend(neutral_cols)
    for col in stripped_cols:
        new_cols.append("team_" + col)
    for col in stripped_cols:
        new_cols.append("opponent_" + col)
    new_cols.append("home")
    ######## End compute new columns

    ######## Build new df
    new_data = []

    for row in season.iterrows():
        home_team = row[1]["home_name"]
        away_team = row[1]["away_name"]

        ### Deal with home team   
        home_row = [home_team, away_team]
        home_row.extend(row[1][neutral_cols])
        home_row.extend(row[1][home_cols])
        home_row.extend(row[1][away_cols])
        home_row.append(1)

        new_data.append(home_row)

        ### Deal with away team
        away_row = [away_team, home_team]
        away_row.extend(row[1][neutral_cols])
        away_row.extend(row[1][away_cols])
        away_row.extend(row[1][home_cols])
        away_row.append(0)

        new_data.append(away_row)

    season_revamped = pd.DataFrame(new_data, columns=new_cols)
    ######## End new df

    # save to file
    season_revamped.to_csv(new_game_data_path + file.split(".")[0] + "_final.csv")
