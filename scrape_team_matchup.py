import json
from pprint import pprint
import pandas as pd
import urllib
import requests
import random
import pickle
from bs4 import BeautifulSoup

base_url = "https://www.teamrankings.com/nba/stat/{}?date={}"
bogus_stat = "average-scoring-margin"

def get_and_format_data(date, fields, verbose=True):
	'''
	Scrapes data and store it in a pickle file
	@params: dates: dates we wish to scrape
			 fields: the fields we want to scrape
	@returns: none
	'''

	headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
	
	
	df = pd.DataFrame()

	for field in fields:
		url = base_url.format(field, date)
		
		r = requests.get(url, headers=headers)
		html_soup = BeautifulSoup(r.text, 'html.parser')
		data_table = html_soup.find_all('table', class_ = "tr-table datatable scrollable")

		if len(data_table) == 0:
			if verbose: print(field)
			continue

		data_table = data_table[0]

		data = data_table.text.split("\n")

		data = [d for d in data if len(d) > 0]

		index = 0
		first_point = True
		reshaped = []
		tmp = None
		for d in data:
			if (len(d) <= 2 and int(d) == index) or index == 0:
				reshaped.append(tmp)
				index += 1
				tmp = []
			else:
				tmp.append(d)
		
		reshaped.append(tmp)
	
		columns = reshaped[1]
		columns = [col + "_" + field if col != "Team" else col for col in columns]


		reshaped = reshaped[2:]
		
		try:
			tmp = pd.DataFrame(reshaped, columns=columns)
		except:
			if verbose:
				print("df failed on:")
				print(field)
				print(columns)
				print("")
			continue

		if tmp.shape[0] != 30:
			if verbose:
				print("wrong shape on:")
				print(field)
				print("")
			continue

		if df.shape[0] == 0:
			df = tmp
		else:
			df = df.merge(tmp, on="Team")

	return df

def add_day(date):
	'''
	Adds a day to the day, which is inputted as a string of the form YYYY-MM-DD"
	'''
	days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

	tokens = date.split("-")
	year = int(tokens[0])
	month = int(tokens[1])
	day = int(tokens[2])

	if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
		days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

	day += 1

	if day > days_in_month[month - 1]:
		day = 1
		month +=1 
		if month > 12:
			month = 1
			year += 1

	day = str(day)
	if len(day) == 1: day = "0" + day

	month = str(month)
	if len(month) == 1: month = "0" + month

	year = str(year)

	return year + "-" + month + "-" + day

def get_fields(date):
	'''
	Gets all the fields we want to scrape.
	Returns a list of fields to insert in the url
	'''
	url = base_url.format(bogus_stat, date)

	headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

	r = requests.get(url, headers=headers)
	html_soup = BeautifulSoup(r.text, 'html.parser')
	fields = html_soup.find_all('div', class_ = "expand-section space-bottom")[0]
	fields = fields.text.split("\n\n")

	to_scrape = []
	for field in fields:
		if len(field.split("\n")) <= 2: continue		
		new_fields = field.split("\n")

		for f in new_fields:
			f = f.lower()
			f = f.replace("-", "")
			f = f.replace("%", "pct")
			f = f.replace(" ", "-")
			f = f.replace("/", "-per-")
			to_scrape.append(f)

	return to_scrape


if __name__ == "__main__":

	start_date = "2007-12-09"
	end_date = "2018-11-12"

	dates = [start_date]
	while start_date != end_date:
		start_date = add_day(start_date)
		dates.append(start_date)

	fields = get_fields(dates[0])

	for date in dates:
		df = get_and_format_data(date, fields, verbose=False)
		print("We are done scraping date {}, and have made a df of shape {}.".format(date, df.shape))
		with open("data/matchups/{}.pkl".format(date), "wb") as f:
			pickle.dump(df, f)

