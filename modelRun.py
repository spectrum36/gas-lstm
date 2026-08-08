import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import math
import time

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

def trainLstm(hDim, lDim, inp, lr, decay, epochs, lossFunc, optim, trainX, trainY, singleRun=True, plot=False):
    
    
    model = lstmModel(input_dim=inp, hidden_dim=hDim, layer_dim=lDim, output_dim=1)

    if optim in ['Adam', 'AdamW']:
        optimFunc = eval(f"torch.optim.{optim}(model.parameters(), lr={lr}, weight_decay={decay})")
    else:
        print("optimizer is not valid")
        return None, None

    h0, c0 = None, None

    #makes it so that epoch loss reports get less common the more epochs there are
    if singleRun==True:
        epochReport = 10 ** (math.floor(math.log10(epochs)) - 1)
    lossList = []
    for epoch in range(epochs):
        model.train()
        optimFunc.zero_grad()

        outputs, h0, c0 = model(trainX, h0, c0)

        loss = lossFunc(outputs, trainY)
        lossList.append(loss.item())
        loss.backward()
        optimFunc.step()

        h0, c0 = h0.detach(), c0.detach()

        if singleRun == True:
            if (epoch + 1) % epochReport == 0 or epoch == 0:
                print(f"epoch [{epoch + 1}/{epochs}], loss: {loss.item():.4f}")
    
    if plot:
        return(model, loss, lossList)
    else:
        return(model, loss)

