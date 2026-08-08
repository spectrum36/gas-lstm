import os
import openmeteo_requests
import pandas as pd
from dotenv import load_dotenv
from influxdb_client_3 import (InfluxDBClient3, InfluxDBError, Point, WritePrecision, WriteOptions, write_client_options)
from geopy.geocoders import Nominatim
import time
from locations import returnLocs
load_dotenv()

def getLoc(addr):
    #notinatim rate limit is 1 request per second, waits one second, tries to request, if succeed return, if fail call again
    time.sleep(1)
    try:
        return geo.geocode(addr).raw
    except Exception as e:
        print(e)
        return getLoc(addr)

geo = Nominatim(user_agent=os.getenv("GEO"))
client = InfluxDBClient3 (host=os.getenv("HOST"), database=os.getenv("WEATHER"), token=os.getenv("TOKEN"))

#get list of 
locList = returnLocs()
locations = {}

for i in locList:
    loc = getLoc(f"{i[1]}, {i[0]}, US")
    locations[i[1].lower()] = [round(float(loc['lat']), 2), round(float(loc['lon']), 2)]

om = openmeteo_requests.Client()

url = "https://archive-api.open-meteo.com/v1/archive"

dfList = []
def callOm(key, value, i):
    if i % 4 == 0:
        time.sleep(60)
    i = i + 1

    try:
        params = {
            "latitude": value[0],
            "longitude": value[1],
            "start_date": "2014-12-20",
            "end_date": "2026-08-07",
            "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum", "snowfall_sum"],
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
        }


        responses = om.weather_api(url, params = params)

        response = responses[0]
        print(f"\nCoordinates: {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation: {response.Elevation()} m asl")
        print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
        print(f"location: {key}")

        daily = response.Daily()
        daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
        daily_precipitation_sum = daily.Variables(3).ValuesAsNumpy()
        daily_rain_sum = daily.Variables(4).ValuesAsNumpy()
        daily_snowfall_sum = daily.Variables(5).ValuesAsNumpy()
        
        #daily_relative_humidity_2m_mean = daily.Variables(6).ValuesAsNumpy()
        #daily_relative_humidity_2m_max = daily.Variables(7).ValuesAsNumpy()
        #daily_relative_humidity_2m_min = daily.Variables(8).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(start = pd.to_datetime(daily.Time(), unit = "s", utc = True), end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True), freq = pd.Timedelta(seconds = daily.Interval()), inclusive = "left")}

        daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
        daily_data["temperature_2m_max"] = daily_temperature_2m_max
        daily_data["temperature_2m_min"] = daily_temperature_2m_min
        daily_data["precipitation_sum"] = daily_precipitation_sum
        daily_data["rain_sum"] = daily_rain_sum
        daily_data["snowfall_sum"] = daily_snowfall_sum
        
        #daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean
        #daily_data["relative_humidity_2m_max"] = daily_relative_humidity_2m_max
        #daily_data["relative_humidity_2m_min"] = daily_relative_humidity_2m_min

        df = pd.DataFrame(data = daily_data)
        df = df.set_index('date')
        client.write(record=df, data_frame_measurement_name=key)
        
        df = df.add_prefix(f"{key}_")
        print("\nDaily data\n", df)
        return df
    except Exception as e:
        print(e)
        if "Hourly" in str(e):
            time.sleep(1800)
        return callOm(key, value, i)
    
    
for key, value in locations.items():
    i = 1
    dfList.append(callOm(key, value, i))

        
dfComb = pd.concat(dfList, axis=1, join='inner')

print(dfComb)

dfComb.to_csv('weather.csv')
