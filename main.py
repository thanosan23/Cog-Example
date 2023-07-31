import torch
import numpy as np

x = np.arange(10, dtype=np.float32)
y = x * 2
y = np.array(y, dtype=np.float32)


x = x.reshape(-1)
y = y.reshape(-1)

x = torch.tensor(x, requires_grad=False)
y = torch.tensor(y, requires_grad=False)

x = torch.unsqueeze(x, 1)
y = torch.unsqueeze(y, 1)

print(x, y)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1)
    def forward(self, x):
        x = x.float()
        x = self.linear1(x)
        return x

model = Model()

optim = torch.optim.Adam(params=model.parameters(), lr=1e-1)
criterion = torch.nn.MSELoss()

for i in range(100):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    print(f"Loss: {loss.item()}")

torch.save(model.state_dict(), "model.pth")

