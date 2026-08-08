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
    hDim: int
    lDim: int
    learn: float
    decay: float
    lossName: str
    reduct: str
    optim: str
    epoch: int
    loss: float
    predict: object
    data: object
    title: str
    lossList: list

def search(seqLen, parameters, paramCount):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def savePredictPlot(predicted, data, title, plotName):
        timesteps = np.arange(len(data))

        plt.figure(figsize=(12,6))
        plt.plot(timesteps, data, label="real change")
        plt.plot(timesteps, predicted, label="predicted chage", linestyle="--")
        plt.title(title)
        plt.xlabel("time step (weeks)")
        plt.ylabel("price change in percent * 100")
        plt.legend()
        plt.savefig(f"{topDir}/{plotName}.png")
        plt.close()

    def saveTrainPlot(lossList, title, plotName):
        timesteps = np.arange(len(lossList))

        plt.figure(figsize=(12,6))
        plt.plot(timesteps, lossList, label="loss")
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(f"loss - {params[5]}")
        plt.legend()
        plt.savefig(f"{topGraphDir}/{plotName}.png")
        plt.close()

    x, y, z, a = sequencesPercent(seqLen, True)

    trainX = torch.tensor(x, dtype=torch.float32)
    trainY = torch.tensor(y[:, None], dtype=torch.float32)
    evalX = torch.tensor(z, dtype=torch.float32)
    evalY = torch.tensor(a[:, None], dtype=torch.float32)
    trainX = trainX.to(torch.device(device))
    trainY = trainY.to(torch.device(device))
    evalX = evalX.to(torch.device(device))
    evalY = evalY.to(torch.device(device))

    count = 0
    lossDict = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for params in parameters:
        count = count + 1
        model = lstmModel(input_dim=len(x[0][0]), hidden_dim=params[0], layer_dim=params[1], output_dim=1)
        model = model.to(device)

        #initalize stuff for the training run
        optimFunc = eval(f"torch.optim.{params[6]}(model.parameters(), lr={params[2]}, weight_decay={params[3]})")
        lossFunc = eval(f'nn.{params[4]}(reduction="{params[5]}")')

        model, trainLoss, lossList = trainLstm(epochs=params[7],
                                    lossFunc=lossFunc,
                                    optimFunc=optimFunc,
                                    trainX=trainX, 
                                    trainY=trainY,
                                    model=model,
                                    singleRun=False
                                    )

        if model == None:
            print("something happened")
            break
        
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

        predicted = predicted.to(torch.device('cpu'))

        plotName = f"{params[0]}-{params[1]}-{params[2]}-{params[3]}-{params[5]}"
        title = f"hDim:{params[0]} | lDim:{params[1]} | learn:{params[2]} | decay:{params[3]} | epochs:{params[7]}\noptim:{params[6]} | loss:{params[4]} | reduct:{params[5]}\n train loss:{trainLoss.item()} | eval loss:{loss.item()}"
        
        #record loss and model configurations for later analysis, as well as plot data
        key = f"{params[6]}-{params[4]}-{params[5]}-{params[7]}"
        if key not in lossDict:
            
            lossDict[key] = lossHolder(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], loss.item(), predicted.detach().numpy(), data, title, lossList)
        else:
            if lossDict[key].loss > loss.item():
                lossDict[key] = lossHolder(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], loss.item(), predicted.detach().numpy(), data, title, lossList)

    msg = "best model configurations:"
    for key, value in lossDict.items():
        names = key.split("-")
        msg = msg + f"\n\n{names[0]}, {names[1]}, {names[2]}\nmodel configuration:\nhDim:{value.hDim} | lDim:{value.lDim} | learn:{value.learn} | decay:{value.decay} | epoch:{value.epoch}\nloss:{value.loss}"
        
        savePredictPlot(value.predict, value.data, value.title, key)
        saveTrainPlot(value.lossList, value.title, key)

        print(msg)

        with open(f"{mainDir}/output.txt", "w") as f:
            f.write(msg)

    

def train():
    torch.manual_seed(42)
    #parameters to be iterated over
    hidden = [64, 80, 90, 100]
    layer = [1, 2, 3]
    learn = [0.005, 0.006, 0.007]
    decay = [0.25, 0.35, 0.45]
    lossName = ["MSELoss", "SmoothL1Loss"]
    reduct = ["mean"]
    optim = ["AdamW"]
    epochs = [500, 1000]

    parameters = itertools.product(hidden, layer, learn, decay, lossName, reduct, optim, epochs)
    paramCount = len(hidden) * len(layer) * len(learn) * len(decay) * len(lossName) * len(reduct) * len(optim) * len(epochs)
    search(10, parameters, paramCount)


train()


