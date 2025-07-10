import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, hidden=64, layers=3):
        super().__init__()

        network_layers = []
        network_layers.append(nn.Linear(in_dim, hidden))
        network_layers.append(nn.Tanh())
        
        for _ in range(layers):
            network_layers.append(nn.Linear(hidden, hidden))
            network_layers.append(nn.Tanh())
            
        network_layers.append(nn.Linear(hidden, out_dim))
        
        self.net = nn.Sequential(*network_layers)

    def forward(self, xyt):
        return self.net(xyt)