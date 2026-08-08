#the sole existence of this is to get a long list of locations for weatherData.py
import pandas as pd
from bs4 import BeautifulSoup
import wikipedia
from io import StringIO
import requests
import os
from dotenv import load_dotenv

def getLocs():
    load_dotenv()

    headers = {
        "User-Agent": os.getenv("WIKI")
            }

    response = requests.get("https://en.wikipedia.org/wiki/List_of_capitals_in_the_United_States", headers=headers)

    text = response.text

    soup = BeautifulSoup(text, 'html.parser')
    
    table = soup.find('table', {"class": "wikitable plainrowheaders sortable"})

    tableDf = pd.read_html(StringIO(str(table)))[0]

    tableDf = tableDf[['State', 'Capital']]

    tableDf = tableDf.drop(tableDf.tail(1).index)

    tableDict = tableDf.to_dict()

    data = {}

    for key, value in tableDict.items():
        for i, p in value.items():
            data.setdefault(i, []).append(p)

    tableList = []
    for key, value in data.items():
        tableList.append(value)

    return tableList

def returnLocs():
    locations = getLocs()
    return locations
