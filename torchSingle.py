from sequencePercent import sequencesPercent
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from modelRun import trainModel

seqLen = 10

x, y, z, a = sequencesPercent(seqLen)

trainX = torch.tensor(x, dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)
evalX = torch.tensor(z, dtype=torch.float32)
evalY = torch.tensor(a[:, None], dtype=torch.float32)

lossFunc =  nn.MSELoss(reduction='mean')

model, loss = trainModel(hDim=8, lDim=1, inp=len(x[0][0]), lr=0.001, decay=0.001, lossFunc=lossFunc, trainX=trainX, trainY=trainY, epochs=1000, optim='AdamW')

h0, c0 = None, None
model.eval()
predicted, _, _ = model(evalX, h0, c0)

data = a.tolist()

loss = lossFunc(predicted, evalY)

print(f"eval loss: {loss.item()}")

original = data
timeSteps = np.arange(len(data))

plt.figure(figsize=(12,6))
plt.plot(timeSteps, original, label="real prices")
plt.plot(timeSteps, predicted.detach().numpy(), label="predicted prices", linestyle="--")
plt.title("my ass trained a model")
plt.xlabel("time step")
plt.ylabel("price change in percent * 1000")
plt.legend()
plt.savefig("plot.png")
