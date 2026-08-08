from sequencePercent import sequencesPercent
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
#seqLen = 4
#seqLen = 2
x, y, z, a = sequencesPercent(seqLen)

trainX = torch.tensor(x, dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)
evalX = torch.tensor(z, dtype=torch.float32)
evalY = torch.tensor(a[:, None], dtype=torch.float32)

model = lstmModel(input_dim=len(x[0][0]), hidden_dim=32, layer_dim=1, output_dim=1)
criterion = nn.MSELoss(reduction="mean")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, weight_decay=0.05)



num_epochs = 1000
h0, c0 = None, None

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs, h0, c0 = model(trainX, h0, c0)

    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()

    h0, c0 = h0.detach(), c0.detach()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"epoch [{epoch + 1}/{num_epochs}], loss: {loss.item():.4f}")

h0, c0 = None, None
model.eval()
predicted, _, _ = model(evalX, h0, c0)

data = a.tolist()

loss = criterion(predicted, evalY)

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
