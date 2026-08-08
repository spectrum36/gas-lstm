from sequencePercent import sequencesPercent
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from modelRun import trainLstm
import time
import os
import itertools

currTime = time.localtime(time.time())
topDir = f'plots/predict-graphs/plots_{currTime[0]}-{currTime[1]}-{currTime[2]}-{currTime[3]:02}{currTime[4]:02}'
topGraphDir = f'plots/train-graphs/plots_{currTime[0]}-{currTime[1]}-{currTime[2]}-{currTime[3]:02}{currTime[4]:02}'

if not os.path.exists('plots'):
    os.mkdir('plots')
if not os.path.exists('plots/predict-graphs'):
    os.mkdir('plots/predict-graphs')
if not os.path.exists('plots/train-graphs'):
    os.mkdir('plots/train-graphs')

seqLen = 10

x, y, z, a = sequencesPercent(seqLen)

trainX = torch.tensor(x, dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)
evalX = torch.tensor(z, dtype=torch.float32)
evalY = torch.tensor(a[:, None], dtype=torch.float32)

hidden = [8, 16, 32, 40]
layer = [1]
learn = [0.001, 0.005, 0.01]
decay = [0.001, 0.005, 0.01, 0.25]
lossName = ["MSELoss", "HuberLoss", "SmoothL1Loss", "L1Loss"]
reduct = ["mean", "sum"]
optim = ["AdamW", "Adam"]
epochs = [1000]
plots = True

parameters = itertools.product(hidden, layer, learn, decay, lossName, reduct, optim, epochs)

for params in parameters:

    lossFunc = eval(f'nn.{params[4]}(reduction="{params[5]}")')

    model, trainLoss, lossList = trainLstm(hDim=params[0], 
                                lDim=params[1], 
                                inp=len(x[0][0]), 
                                lr=params[2], 
                                decay=params[3], 
                                lossFunc=lossFunc, 
                                trainX=trainX, 
                                trainY=trainY, 
                                epochs=params[7], 
                                optim=params[6], 
                                singleRun=False,
                                plot=plots
                                )

    if model == None:
        print("something happened")
        break
 
    #optional loss graphs
    if plots == True:
        if not os.path.exists(topGraphDir):
            os.mkdir(topGraphDir)

        plotName = f"{params[0]}-{params[1]}-{params[2]}-{params[3]}-{params[6]}-{params[4]}-{params[5]}"

        timeSteps = np.arange(len(lossList))

        plt.figure(figsize=(12,6))
        plt.plot(timeSteps, lossList, label="loss")
        plt.title(f"hDim:{params[0]} | lDim:{params[1]} | learn:{params[2]} | decay:{params[3]} \noptim:{params[6]} | loss:{params[4]}")
        plt.xlabel("epoch")
        plt.ylabel(f"loss - {params[5]}")
        plt.legend()
        plt.savefig(f"{topGraphDir}/{plotName}.png")
        plt.close()

    

    h0, c0 = None, None
    model.eval()
    predicted, _, _ = model(evalX, h0, c0)

    data = a.tolist()

    loss = lossFunc(predicted, evalY)

    print(f"hDim:{params[0]} | lDim:{params[1]} | learn:{params[2]} | decay:{params[3]}")
    print(f"optim:{params[6]} | loss:{params[4]} | reduct:{params[5]} | epochs:{params[7]}")
    print(f"eval loss: {loss.item()} | last train loss: {trainLoss.item()}\n")

    original = data
    timeSteps = np.arange(len(data))

    #make dirs to hold plots
    plotsDir = f'{topDir}/{params[4]}-{params[6]}'

    if not os.path.exists(topDir):
        os.mkdir(topDir)
    if not os.path.exists(plotsDir):
        os.mkdir(plotsDir)

    plotName = f"{params[0]}-{params[1]}-{params[2]}-{params[3]}-{params[5]}"

    plt.figure(figsize=(12,6))
    plt.plot(timeSteps, original, label="real prices")
    plt.plot(timeSteps, predicted.detach().numpy(), label="predicted prices", linestyle="--")
    plt.title(f"hDim:{params[0]} | lDim:{params[1]} | learn:{params[2]} | decay:{params[3]} | epochs:{params[7]}\noptim:{params[6]} | loss:{params[4]} | reduct:{params[5]}\n train loss:{trainLoss.item()} | eval loss:{loss.item()}")
    plt.xlabel("time step (weeks)")
    plt.ylabel("price change in percent * 100")
    plt.legend()
    plt.savefig(f"{plotsDir}/{plotName}.png")
    plt.close()
