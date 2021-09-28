import torch
from torch._C import parse_schema
import torch.nn as nn
from .ParallelNeuralIntegral import ParallelNeuralIntegral

def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

class PostiveIntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(PostiveIntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1)) + 1.

class IncConcaveNN(nn.Module):
    # increasing concave neural network
    def __init__(self, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super(IncConcaveNN, self).__init__()

        self.InnerIntegrand = InnerUMNN(in_d, hidden_layers, nb_steps=nb_steps, dev=dev)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        # It will output the scaling and offset factors.
        self.net = nn.Sequential(*self.net)
        self.device = dev
        self.nb_steps = nb_steps
    
    def forward(self, x, h):
        x0 = torch.zeros(x.shape).to(self.device)
        out = self.net(h)
        offset = out[:, [0]]
        scaling = torch.exp(out[:, [1]])

        OuterIntegralOut = scaling*ParallelNeuralIntegral.apply(x0, x, self.InnerIntegrand,
                                                            _flatten(self.InnerIntegrand.parameters()), h, self.nb_steps) + offset
        return OuterIntegralOut


class InnerUMNN(nn.Module):
    def __init__(self, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super(InnerUMNN, self).__init__()
        
        self.device = dev
        self.nb_steps = nb_steps
        self.Integrand = PostiveIntegrandNN(in_d, hidden_layers)

        
    def forward(self, x, h):
        xmax = torch.max(x)+10
        xfinal = torch.zeros(x.shape).to(self.device)+xmax
        x = x.to(self.device)
        h = h.to(self.device)
        OuterIntegralOut = ParallelNeuralIntegral.apply(x, xfinal, self.Integrand,
                                                        _flatten(self.Integrand.parameters()), h, self.nb_steps)
        
        return  OuterIntegralOut