# client.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy

class Client:
    """Client for federated learning."""
    def __init__(self, client_id, model, dataset, device="cpu"):
        self.client_id = client_id
        self.model = deepcopy(model)
        self.dataset = dataset
        self.device = device

    def train(self, epochs=1, batch_size=64, lr=0.001):
        """Train locally on client dataset."""
        self.model.to(self.device)
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

    def get_parameters(self):
        """Return model parameters as state_dict."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_parameters(self, params):
        """Set model parameters from state_dict."""
        self.model.load_state_dict(params)
