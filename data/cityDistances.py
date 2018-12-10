#Answering the question: Which NBA teams have the most distance between them?
#The WolframAlpha API returns the miles between two locations.
# Credit to https://github.com/myzeiri/Distances-Between-Cities

import urllib.request, urllib.error, urllib.parse
from xml.etree import cElementTree as ET
import pandas as pd
appID = "GUJ6PG-PK9RG8VL6K" #specific to this program


def miles_between(city1, city2):
	query = "Distance+between+" + city1 + "+and+" + city2
	url = "http://api.wolframalpha.com/v2/query?input="+query+"&appid="+str(appID)+"&includepodid=Result"
	s = urllib.request.urlopen(url)
	contents = s.read()
	root = ET.fromstring(contents)
	try:
		distanceStr = root[0][0][1].text 
		print(distanceStr)
		#ignoring the " miles" at the end of the plaintext result
		return float(distanceStr[0:distanceStr.index(" ")])
	except:
		return -1.

#Note that you can pass in the same conference twice
def farthest_cities_within(conference1, conference2):
	df = pd.DataFrame({'City1' : [], 'City2' : [], 'Distance' : []})
	while len(conference1) > 0:
		last = conference1.pop()
		print(last)
		for city in conference2:
			d = miles_between(last, city)
			df = df.append({'City1': last, 'City2': city, 'Distance': d}, ignore_index=True)
	df.to_csv('distances.csv', index=False, header=False)

atlantic = ["Boston", "New+York+City", "Philadelphia", "Toronto"]
central = ["Chicago", "Cleveland", "Auburn+Hills", "Indianapolis", "Milwaukee"]
southeast = ["Atlanta", "Charlotte", "Miami", "Orlando", "Washington+DC"]

northwest = ["Denver", "Minneapolis", "Oklahoma+City", "Portland", "Salt+Lake+City", "Seattle"]
pacific = ["Oakland", "LA", "Phoenix", "Sacramento"]
southwest = ["Dallas", "Houston", "Memphis", "New+Orleans", "San+Antonio"]

east = atlantic + central + southeast
west = northwest + pacific + southwest

total = east + west

farthest_cities_within(total, total)