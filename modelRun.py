import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import math
import time

def trainLstm(epochs, lossFunc, optimFunc, trainX, trainY, model, singleRun=True):
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
    
    return(model, loss, lossList)
    return(model, loss)

