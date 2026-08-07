import pandas as pd
import pyarrow as pa
import os
from dotenv import load_dotenv
from influxdb_client_3 import (
  InfluxDBClient3, InfluxDBError, Point, WritePrecision,
  WriteOptions, write_client_options)

def getDfList(name, startDate, client):
    dfList = []

    for i in name:
        n = client.query(f"SELECT * FROM '{i}' WHERE time <= '{startDate}-07-20T00:00:00.000Z' AND time >= '{startDate - 1}-07-21T00:00:00.000Z';")
        
        balls = {}

        for batch in n.to_batches():
            d = batch.to_pydict()
            
            for series, value, time, typ, units in zip(d['series'], d['value'], d['time'], d['type'], d['units']):
                time = str(time)
                time = time.partition(" 00:")[0]
                balls.setdefault(f'{typ}_{series}_{units}', []).append([time, value])

        for key,value in balls.items():
            df = pd.DataFrame(value, columns=["date", f"{key}"])
            df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
            fullRange = pd.date_range(start=f"{startDate - 1}-7-21", end=f"{startDate}-7-20", freq='D')
            df = df.reindex(fullRange)
            df = df.sort_index(ascending=False)
            dfList.append(df)
    
    return(dfList)

def getPriceDf (startDate, endDate, infill=True, debug=False):
    load_dotenv()

    client = InfluxDBClient3 (
            host=os.getenv("HOST"),
            database=os.getenv("DATABASE"),
            token=os.getenv("TOKEN"))

    n = client.query("show tables")

    name = []

    for i, p in zip(n['table_schema'], n['table_name']):
        if str(i) == "iox":
            name.append(str(p))

    dfListMaster = []
    for i in range(endDate, startDate, -1):
        dfList = getDfList(name, i, client)
        df = pd.concat(dfList, axis=1)
        dfListMaster.append(df)
        if debug == True:
            print(df)
    df3 = pd.concat(dfListMaster, axis=0, join='inner')

    if infill == True:
        interpolate = ['spotPrice', 'stock']
        for i in interpolate:
            prefixCols = df3.filter(like=f"{i}_")
            df3[prefixCols.columns] = prefixCols.interpolate(method='linear', limit_direction='both')
    
        prefixCols = df3.filter(like="weekly")
        df3[prefixCols.columns] = prefixCols.bfill()
        prefixCols = df3.filter(like="weekly")
        df3[prefixCols.columns] = prefixCols.ffill()

    return(df3)

