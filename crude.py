from dotenv import load_dotenv
import os
import requests
import time
import datetime
import json
from influxdb_client_3 import (
  InfluxDBClient3, InfluxDBError, Point, WritePrecision,
  WriteOptions, write_client_options)

load_dotenv()

client = InfluxDBClient3 (
        host=os.getenv("HOST"),
        database=os.getenv("DATABASE"),
        token=os.getenv("TOKEN"))

apiKey = "&api_key=" + os.getenv("KEY")

def getLength(url):
    d = requests.get(url)
    return(int(d.json()['response']['total']))

series = {}
product = {}
area = {}

def writeToDb(url, wdGaster):
    length = getLength(url + "&offset=0&length=1" + apiKey) 
    offset = 0
    for _ in range((length // 5000) + 1):
        points = []
        d = requests.get(url + "&offset=" + str(offset) + "&length=5000" + apiKey)
        for i in d.json()['response']['data']:
            if i['series'] not in series:
                series[i['series']] = i['series-description']
            if i['product'] not in product:
                product[i['product']] = i['product-name']
            if i['duoarea'] not in area:
                area[i['duoarea']] = i['area-name']
            if i['value'] != None:
                points.append(Point(i['product-name']).tag("area", i['duoarea']).tag("series", i['series']).tag("units", i['units']).tag("type", wdGaster).field("value", float(i['value'])).time(i['period']))
        offset = offset + 5000
        client.write(points, write_precision='s')

urls = {
        "https://api.eia.gov/v2/petroleum/pri/spt/data/?frequency=daily&data[0]=value&start=2015-01-01&sort[0][column]=period&sort[0][direction]=desc": "spotPrice", 
        "https://api.eia.gov/v2/petroleum/stoc/wstk/data/?frequency=weekly&data[0]=value&start=2015-01-01&sort[0][column]=period&sort[0][direction]=desc": "stock", 
        "https://api.eia.gov/v2/petroleum/pnp/wprodrb/data/?frequency=weekly&data[0]=value&start=2015-01-01&sort[0][column]=period&sort[0][direction]=desc": "weeklyProduction",
        "https://api.eia.gov/v2/petroleum/move/wkly/data/?frequency=weekly&data[0]=value&start=2015-01-01&sort[0][column]=period&sort[0][direction]=desc": "weeklyImportExport",
        "https://api.eia.gov/v2/petroleum/cons/wpsup/data/?frequency=weekly&data[0]=value&start=2015-01-01&sort[0][column]=period&sort[0][direction]=desc": "weeklyUsSupplied",
        "https://api.eia.gov/v2/petroleum/pnp/wiup/data/?frequency=weekly&data[0]=value&start=2015-01-01&sort[0][column]=period&sort[0][direction]=desc": "weeklyInputsUtilization",
        "https://api.eia.gov/v2/petroleum/pri/gnd/data/?frequency=weekly&data[0]=value&start=2015-01-01&sort[0][column]=period&sort[0][direction]=desc": "weeklyRetailPrice"
        }
for key, value in urls.items():
    writeToDb(key, value)

with open("eiaPetroleumSeries.json", "w", encoding="utf-8") as file:
    json.dump(series, file)

with open("eiaPetroleumProduct.json", "w", encoding="utf-8") as file:
    json.dump(product, file)

with open("eiaPetroleumArea.json", "w", encoding="utf-8") as file:
    json.dump(area, file)
