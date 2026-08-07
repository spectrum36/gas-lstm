import pandas as pd
from priceDf import getPriceDf
import numpy as np
import os

def saveSeq():
    nofillDf = getPriceDf(2015, 2026, infill=False)
    #infillDf = getPriceDf(2015, 2026, infill=True)

    nofillDf.to_csv('nofillDf.csv')                       #infillDf.to_csv('infillDf.csv')

def createSequences(data, seqLen, multi):
    xs, ys = [], []

    for i in range(len(data) - seqLen):
        if multi == True:
            x = data[i:(i + seqLen)]
        else:
            x = []
            z = data[i:(i + seqLen)]
            for p in z:
                x.append(p[1])
        y = data[i + seqLen][0]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def getSequence(df):
    indexDate = "2016-01-01"

    gas = "EMM_EPMR_PTE_YORD_DPG"
    gasCols = df.filter(like=gas)
    column = gasCols.columns
    gasCols = gasCols.dropna()
    mask = (gasCols.index >= indexDate)
    gasCols = gasCols.loc[mask]
    gasCols = gasCols.pct_change(periods=-1)
    gasCols = gasCols.drop(gasCols.index[-1])
    
    #print(gasCols)

    spotPrice = ["RWTC", "RBRTE", "WCRSTUS1", "WDISTUS1", "WTTIMUS2", "EER_EPMRU_PF4_Y35NY_DPG", "EER_EPD2DC_PF4_Y05LA_DPG"] 
    spotDfs = []

    for i in spotPrice:
        spotCol = df.filter(like=i)
        spotCol = spotCol.sort_index(ascending=True)
        spotCol = spotCol.rolling(window='7D').mean()
        spotCol = spotCol.resample("W-MON").mean()
        #spotCol = spotCol.sort_index(ascending=False)
        spotCol = spotCol.pct_change(periods=-1)
        spotCol = spotCol.sort_index(ascending=False)
        #print(spotCol)
        spotDfs.append(spotCol)

    spotCols = pd.concat(spotDfs, axis=1)

    combCols = gasCols.merge(spotCols, how="inner", left_index=True, right_index=True)
    combCols = combCols.sort_index(ascending=True)
    
    #multiply every value for scaling reasons
    combCols = combCols.mul(100)
    
    #generate seasonality columns
    dates = pd.date_range(start="2016-1-1", end="2026-1-1", freq="D")

    dateDf = pd.DataFrame({"date": dates})
    print(dateDf)
    dateDf = dateDf.set_index("date")

    dateDf["dayOfYear"] = dateDf.index.day_of_year

    dateDf["sine"] = np.sin(2 * np.pi * dateDf["dayOfYear"] / 365.25)

    dateDf["cosine"] = np.cos(2 * np.pi * dateDf["dayOfYear"] / 365.25)

    dateDf = dateDf.drop(columns=['dayOfYear'])

    combCols = combCols.merge(dateDf, how="inner", right_index=True, left_index=True)

    print(combCols)
    
    trainMask =  (combCols.index >= "2016-01-01") & (combCols.index < "2025-01-01")
    evalMask = (combCols.index >= "2025-01-01") & (combCols.index <= "2026-01-01")

    trainSet = combCols.loc[trainMask]
    evalSet = combCols.loc[evalMask]
    
    trainSet = trainSet.reset_index(drop=True)
    evalSet = evalSet.reset_index(drop=True)

    def getList(df):

        dfSet = df.to_dict()
        dfDict = {}
        dfList = []

        for i in dfSet:
            for key, value in dfSet[i].items():
                dfDict.setdefault(key, []).append(value)

        for key, value in dfDict.items():
            i = [x for x in value]
            dfList.append(i)

        return dfList

    trainList = getList(trainSet)
    evalList = getList(evalSet)

    return(trainList, evalList)



def sequencesPercent(seqLen, multi=False):
    
    if os.path.exists('nofillDf.csv'):
        nofillDf = pd.read_csv('nofillDf.csv')
    else:
        saveSeq()
        nofillDf = pd.read_csv('nofillDf.csv')

    nofillDf = nofillDf.rename(columns={nofillDf.columns[0]: 'date'}) 
    
    nofillDf['date'] = pd.to_datetime(nofillDf['date'])

    nofillDf = nofillDf.set_index('date')

    seq, evalSeq = getSequence(nofillDf)

    seqTrainX, seqTrainY = createSequences(seq, seqLen, multi)

    seqEvalX, seqEvalY = createSequences(evalSeq, seqLen, multi)
    
    #print(seqTrainX)
    #print(seqEvalX)

    return(seqTrainX, seqTrainY, seqEvalX, seqEvalY)

#sequencesPercent(10, multi=True)
