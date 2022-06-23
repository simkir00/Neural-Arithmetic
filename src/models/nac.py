import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.models.default import GeneralModel


class NAC_Cell(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W_hat = Parameter(Tensor(output_dim, input_dim))
        self.M_hat = Parameter(Tensor(output_dim, input_dim))
        self.W = Parameter(torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat))

        self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.W_hat)
        nn.init.xavier_uniform_(self.M_hat)

    def forward(self, X):
        return F.linear(X, self.W, self.bias)


class NAC(GeneralModel):
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_dim):
        super().__init__(input_dim, output_dim, num_hidden_layers, hidden_dim)
        self.model = self.build_model()

    def layer(self, input_dim, output_dim):
        layer = [NAC_Cell(input_dim, output_dim)]
        return layer

    def name(self):
        s = "NAC"
        return s
