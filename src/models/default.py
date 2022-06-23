import torch
import torch.nn as nn


class GeneralModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim

        self.model = None

    def layer(self, input_dim, output_dim):
        raise NotImplementedError

    def build_model(self):
        layers = self.layer(self.input_dim, self.hidden_dim)
        for _ in range(self.num_hidden_layers - 1):
            layers += self.layer(self.hidden_dim, self.hidden_dim)
        layers += self.layer(self.hidden_dim, self.output_dim)
        res_model = nn.Sequential(*layers)
        return res_model

    def forward(self, X):
        res = self.model(X)
        return res

    def name(self):
        raise NotImplementedError

    def __str__(self):
        return self.name()
