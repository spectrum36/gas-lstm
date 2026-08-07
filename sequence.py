import pandas as pd
from priceDf import getPriceDf
from dateRange import getDateRange
import numpy as np

def getSequence(df):
    gas = "EMM_EPMR_PTE_YORD_DPG"
    gasCols = df.filter(like=gas)
    column = gasCols.columns
    crudeCols = df.filter(like="RWTC")
    gasCols = gasCols.dropna()
    gasDates = gasCols.index.tolist()

    sequence = []
    
    train = []
    evaluate = []
    evalSeq = []

    for i in gasDates:
        x = str(i)

        if "2025" in x:
            evaluate.append(i)
        elif "2026" in x:
            pass
        else:
            train.append(i)

    for i in train:
        date = str(i)
        date = date.partition(" 00:")[0]
        startDate = getDateRange(date, 7)

        mask = (crudeCols.index > startDate) & (crudeCols.index <= date)

        crude = crudeCols.loc[mask]
        crude = crude.dropna()
        crude = crude.reset_index(drop=True)
        #crude = crude.tolist()

        lis = []

        dickt = crude.to_dict()

        for key, value in dickt.items():
            lis2 = value
            for p, b in lis2.items():
                lis.append(b)

        x = 0
        for p in lis:
            x = x + p

        x = round(x / len(lis), 2)

        col = gasCols.loc[gasCols.index == i, column]
        col = col.reset_index(drop=True)

        dickt = col.to_dict()

        for key, value in dickt.items():
            lis2 = value
            for p, b in lis2.items():
                y = b

        sequence.append([x,y])


    for i in evaluate:
        date = str(i)
        date = date.partition(" 00:")[0]
        
        startDate = getDateRange(date, 7)
        mask = (crudeCols.index > startDate) & (crudeCols.index <= date)

        crude = crudeCols.loc[mask]
        crude = crude.dropna()
        crude = crude.reset_index(drop=True)
        #crude = crude.tolist()

        lis = []

        dickt = crude.to_dict()

        for key, value in dickt.items():
            lis2 = value
            for p, b in lis2.items():
                lis.append(b)

        x = 0
        for p in lis:
            x = x + p

        x = round(x / len(lis), 2)


        col = gasCols.loc[gasCols.index == i, column]
        col = col.reset_index(drop=True)

        dickt = col.to_dict()

        for key, value in dickt.items():
            lis2 = value
            for p, b in lis2.items():
                y = b
        evalSeq.append([x, y])

    return(sequence, evalSeq)

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
        y = data[i + seqLen][1]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def sequences(seqLen, multi=False):
    
    #infillDf = getPriceDf(2016, 2026)
    nofillDf = getPriceDf(2016, 2026, infill=False)

    seq, evalSeq = getSequence(nofillDf)

    seqTrainX, seqTrainY = createSequences(seq, seqLen, multi)

    seqEvalX, seqEvalY = createSequences(evalSeq, seqLen, multi)
    
    #print(seqTrainX)

    return(seqTrainX, seqTrainY, seqEvalX, seqEvalY)

#sequences(10, True)
