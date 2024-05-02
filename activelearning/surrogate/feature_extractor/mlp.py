import torch.nn as nn
import torch
from gflownet.utils.common import set_float_precision

ACTIVATION_KEY = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
}


class Identity(nn.Module):
    def __init__(self, n_output):
        super(Identity, self).__init__()
        self.n_output = n_output

    def get_features(self, x, **kwargs):
        return x

    def forward(self, x, **kwargs):
        return self.get_features(x)


class MLP(torch.nn.Sequential):
    def __init__(self, n_input, n_hidden=[32, 64], n_output=2, float_precision=32):
        self.float = set_float_precision(float_precision)
        super(MLP, self).__init__()
        self.add_module(
            "linear1", torch.nn.Linear(n_input, n_hidden[0], dtype=self.float)
        )
        self.add_module("relu1", torch.nn.ReLU())
        for i in range(len(n_hidden) - 1):
            self.add_module(
                "linear%i" % (i + 2),
                torch.nn.Linear(n_hidden[i], n_hidden[i + 1], dtype=self.float),
            )
            self.add_module("relu%i" % (i + 2), torch.nn.ReLU())
        self.add_module(
            "linear_out", torch.nn.Linear(n_hidden[-1], n_output, dtype=self.float)
        )
        self.add_module("layer_norm", nn.LayerNorm(n_output))

        self.n_output = n_output
