import torch
from torch import nn
from torch import from_numpy, tensor
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 8)
        self.l2 = nn.Linear(8, 8)
        self.l3 = nn.Linear(8, 4)
        self.l4 = nn.Linear(4, 2)
        self.l5 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Model()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)


raw_data = np.loadtxt('./qsar_aquatic_toxicity.csv', delimiter=';', dtype=np.float32)
x_tensor = torch.from_numpy(raw_data[:, :-1])
y_tensor = torch.from_numpy(raw_data[:, [-1]])
x_data = x_tensor[100:,]
y_data = y_tensor[100:]
x_data_ts = x_tensor[0:100,]
y_data_ts = y_tensor[0:100]

mu = x_data.mean()
sigma = x_data.std()
x_normalized = (x_data-mu)/sigma
x_ts_normalized = (x_data_ts-mu)/sigma

for epoch in range(1000):
    y_pred = model(x_normalized.float())
    loss = criterion(y_pred, y_data.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print(f'Loss: {loss.item()}')
y_pred_ts = model(x_ts_normalized.float())
loss_ts = criterion(y_pred_ts, y_data_ts.float())
print(f'Testing Result | Loss: {loss_ts.item()} ')
plt.scatter(y_data_ts.detach().numpy(), y_pred_ts.detach().numpy())
plt.xlim(0,15)
plt.ylim(0,15)
plt.ylabel('True y')
plt.xlabel('Predicted y')
plt.title('Training data')
print(f'Original Data: {raw_data.shape}')
print(f'x_data: {x_data.shape} y_data: {y_data.shape}\nx_data_ts: {x_data_ts.shape} y_data_ts: {y_data_ts.shape}')
