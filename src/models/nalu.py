import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.models.default import GeneralModel
from src.models.nac import NAC_Cell


class NALU_Cell(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.eps = 1e-10

        self.G = Parameter(Tensor(self.output_dim, self.input_dim))
        self.NAC_cell_1 = NAC_Cell(self.input_dim, self.output_dim)
        self.NAC_cell_2 = NAC_Cell(self.input_dim, self.output_dim)

        self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.G)

    def forward(self, X):
        a = self.NAC_cell_1(X)
        m = torch.exp(self.NAC_cell_2(torch.log(torch.abs(X) + self.eps)))
        g = torch.sigmoid(F.linear(X, self.G, self.bias))
        return a * g + (1 - g) * m


class NALU(GeneralModel):
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_dim):
        super(NALU, self).__init__(input_dim, output_dim, num_hidden_layers, hidden_dim)
        self.model = self.build_model()

    def layer(self, input_dim, output_dim):
        layer = [NALU_Cell(input_dim, output_dim)]
        return layer

    def name(self):
        s = "NALU"
        return s