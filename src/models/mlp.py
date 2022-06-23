import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.default import GeneralModel
from src.models.utils import string_to_activation_func


class MLP(GeneralModel):
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_dim, activation="relu"):
        super().__init__(input_dim, output_dim, num_hidden_layers, hidden_dim)
        self.activation = string_to_activation_func(activation)
        self.model = self.build_model()

    def layer(self, input_dim, output_dim):
        if (output_dim == self.output_dim) or (self.activation is None):
            layer = [nn.Linear(input_dim, output_dim)]
        else:
            layer = [nn.Linear(input_dim, output_dim), self.activation()]
        return layer

    def name(self):
        s = "MLP"
        return s


if __name__ == "__main__":
    print("Hello, world!")
