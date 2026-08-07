from sequencePercent import sequencesPercent
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

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

seqLen = 10

x, y, z, a = sequencesPercent(seqLen, multi=True)

trainX = torch.tensor(x, dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)
evalX = torch.tensor(z, dtype=torch.float32)
evalY = torch.tensor(a[:, None], dtype=torch.float32)

outs = []

def trainModel(hDim, lDim, lr, decay):
    model = lstmModel(input_dim=len(x[0][0]), hidden_dim=hDim, layer_dim=lDim, output_dim=1)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

    num_epochs = 300
    h0, c0 = None, None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs, h0, c0 = model(trainX, h0, c0)

        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()

        h0, c0 = h0.detach(), c0.detach()

    h0, c0 = None, None
    model.eval()
    predicted, _, _ = model(evalX, h0, c0)

    data = a.tolist()

    loss = criterion(predicted, evalY)

    print(f"hidden dim:{hDim}, layer dim:{lDim}, learn rate:{lr}, weight decay:{decay}")
    print(f"eval loss: {loss.item():.4f}")

    outs.append(f"hidden dim:{hDim}, layer dim:{lDim}, learn rate:{lr}, weight decay:{decay}\neval loss: {loss.item():.4f}\n\n")

    original = data
    timeSteps = np.arange(len(data))

    data = a.tolist()

    plt.figure(figsize=(12,6))
    plt.plot(timeSteps, original, label="real prices")
    plt.plot(timeSteps, predicted.detach().numpy(), label="predicted prices", linestyle="--")
    plt.title(f" gas price lstm: hidden:{hDim}, layer:{lDim}, learn rate:{lr}, weight decay:{decay}, loss:{loss.item():.4f}")
    plt.xlabel("time step")
    plt.ylabel("price change in percent * 100")
    plt.legend()
    plt.savefig(f"plots/{hDim}-{lDim}-{lr}-{decay}.png")
    plt.close()

hDim = [8, 16, 32, 64, 128]
lDim = [1, 2]
lr = [0.005, 0.01, 0.02]
decay = [0.025, 0.05, 0.1, 0.25]

for r in decay:
    for q in lr:
        for i in hDim:
            for p in lDim:
                trainModel(i, p, q, r)

if os.path.exists("plots/output.txt"):
    os.remove("plots/output.txt")
    
file = open("plots/output.txt", "a")
for i in outs:
    file.write(i)

    
