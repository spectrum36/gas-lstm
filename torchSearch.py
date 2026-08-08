from sequencePercent import sequencesPercent
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from modelRun import trainLstm
import time
import os
import itertools
import datetime
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class lstmModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(lstmModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out, hn, cn

@dataclass
class lossHolder:
    hDim: hDim
    lDim: lDim
    learn: learn
    decay: decay
    lossName: lossName
    reduct: reduct
    optim: optim
    epoch: epoch
    loss: loss
    predict: predict
    data: data
    title: title
    timesteps: timesteps

def savePredictPlot(predicted, data, timesteps, title, plotsDir, plotName):
    plt.figure(figsize=(12,6))
    plt.plot(timeSteps, data, label="real change")
    plt.plot(timeSteps, predicted, label="predicted chage", linestyle="--")
    plt.title(title)
    plt.xlabel("time step (weeks)")
    plt.ylabel("price change in percent * 100")
    plt.legend()
    plt.savefig(f"{plotsDir}/{plotName}.png")
    plt.close()

startTime = time.time()
currTime = time.localtime(startTime)

top = f'output'
mainDir = f'{top}/multi-plots_{currTime[0]}-{currTime[1]}-{currTime[2]}-{currTime[3]:02}{currTime[4]:02}'
topDir = f'{mainDir}/predict-plots'
topGraphDir = f'{mainDir}/training-plots'

paths = [top , mainDir, topDir, topGraphDir]

for i in paths:
    if not os.path.exists(i):
        os.mkdir(i)

seqLen = 10

x, y, z, a = sequencesPercent(seqLen, False)


trainX = torch.tensor(x, dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)
evalX = torch.tensor(z, dtype=torch.float32)
evalY = torch.tensor(a[:, None], dtype=torch.float32)
trainX = trainX.to(torch.device(device))
trainY = trainY.to(torch.device(device))
evalX = evalX.to(torch.device(device))
evalY = evalY.to(torch.device(device))

hidden = [8, 16, 32, 40, 48]
layer = [1]
learn = [0.001, 0.005, 0.01]
decay = [0.001, 0.005, 0.01, 0.25]
lossName = ["MSELoss", "HuberLoss", "SmoothL1Loss"]
reduct = ["mean", "sum"]
optim = ["AdamW", "Adam"]
epochs = [100, 500, 1000]
plots = True

parameters = itertools.product(hidden, layer, learn, decay, lossName, reduct, optim, epochs)

paramCount = len(hidden) * len(layer) * len(learn) * len(decay) * len(lossName) * len(reduct) * len(optim) * len(epochs)
count = 0
lossDict = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for params in parameters:
    count = count + 1
    model = lstmModel(input_dim=len(x[0][0]), hidden_dim=params[0], layer_dim=params[1], output_dim=1)
    model = model.to(device)

    #initalize stuff for the training run
    if params[6] in ['Adam', 'AdamW']:
        optimFunc = eval(f"torch.optim.{params[6]}(model.parameters(), lr={params[2]}, weight_decay={params[3]})")
    else:
        print("optimizer is not valid")
        break

    lossFunc = eval(f'nn.{params[4]}(reduction="{params[5]}")')


    model, trainLoss, lossList = trainLstm(epochs=params[7],
                                lossFunc=lossFunc,
                                optimFunc=optimFunc,
                                trainX=trainX, 
                                trainY=trainY,
                                model=model,
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

    seconds = int(time.time()) - int(startTime)

    print(f"test:{count}/{paramCount} | time elapsed:{str(datetime.timedelta(seconds=seconds))}")
    print(f"hDim:{params[0]} | lDim:{params[1]} | learn:{params[2]} | decay:{params[3]}")
    print(f"optim:{params[6]} | loss:{params[4]} | reduct:{params[5]} | epochs:{params[7]}")
    print(f"eval loss: {loss.item()} | last train loss: {trainLoss.item()}\n")

    original = data
    timeSteps = np.arange(len(data))

    #make dirs to hold plots
    plotsDir = f'{topDir}/{params[4]}-{params[6]}'

    if not os.path.exists(plotsDir):
        os.mkdir(plotsDir)

    predicted = predicted.to(torch.device('cpu'))

    plotName = f"{params[0]}-{params[1]}-{params[2]}-{params[3]}-{params[5]}"
    title = f"hDim:{params[0]} | lDim:{params[1]} | learn:{params[2]} | decay:{params[3]} | epochs:{params[7]}\noptim:{params[6]} | loss:{params[4]} | reduct:{params[5]}\n train loss:{trainLoss.item()} | eval loss:{loss.item()}"
    
    savePredictPlot(predicted.detach().numpy(), data, timeSteps, title, plotsDir, plotName)

    #record loss and model configurations for later analysis, as well as plot data
    key = f"{params[6]}-{params[4]}-{params[5]}"
    if key not in lossDict:
        lossDict[key] = lossHolder(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], loss.item(), predicted.detach().numpy(), data, title, timeSteps)
    else:
        if lossDict[key].loss > loss.item():
            lossDict[key] = lossHolder(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], loss.item(), predicted.detach().numpy(), data, title, timeSteps)

    
out = f'{mainDir}/output'
os.mkdir(out)
msg = "best model configurations:"
for key, value in lossDict.items():
    names = key.split("-")
    msg = msg + f"\n\n{names[0]}, {names[1]}, {names[2]}\nmodel configuration:\nhDim:{value.hDim} | lDim:{value.lDim} | learn:{value.learn} | decay:{value.decay} | epoch:{value.epoch}\nloss:{value.loss}"
    
    savePredictPlot(value.predict, value.data, value.timesteps, value.title, out, key)
    
print(msg)

with open(f"{out}/output.txt", "w") as f:
    f.write(msg)
