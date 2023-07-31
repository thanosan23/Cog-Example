from cog import BasePredictor
from typing import Any
import torch
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1)
    def forward(self, x):
        x = self.linear1(x)
        return x


class Predictor(BasePredictor):
    def setup(self):
        self.model = Model()
        self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()

    def predict(self, x : float) -> Any:
        with torch.no_grad():
            arr = np.array([x], dtype=np.float64)
            inp = torch.from_numpy(arr).float()
            output = self.model(inp.float())
            output = output.item()
        return output
