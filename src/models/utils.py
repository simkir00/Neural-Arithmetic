import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION_FUNCTIONS = ["ReLU", "LeakyReLU", "ReLU6", "RReLU", "GELU", "ELU",
                        "Sigmoid", "LogSigmoid", "Hardsigmoid",
                        "Tanh", "Tanhshrink", "Hardtanh",
                        "Hardswish", "Softplus", "Mish", "Hardshrink"]

ARITHMETIC_OPERATIONS = {
    "a + b": lambda a, b: a + b,
    "a - b": lambda a, b: a - b,
    "a * b": lambda a, b: a * b,
    "a / b": lambda a, b: a / b,
    "a^2": lambda a, _: a ** 2,
    "sqrt(a)": lambda a, _: np.sqrt(a),
}


def string_to_activation_func(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    elif name == "leakyrelu":
        return nn.LeakyReLU
    elif name == "relu6":
        return nn.ReLU6
    elif name == "rrelu":
        return nn.RReLU
    elif name == "gelu":
        return nn.GELU
    elif name == "elu":
        return nn.ELU
    elif name == "sigmoid":
        return nn.Sigmoid
    elif name == "logsigmoid":
        return nn.LogSigmoid
    elif name == "hardsigmoid":
        return nn.Hardsigmoid
    elif name == "tanh":
        return nn.Tanh
    elif name == "tanhshrink":
        return nn.Tanhshrink
    elif name == "hardtanh":
        return nn.Hardtanh
    elif name == "hardswish":
        return nn.Hardswish
    elif name == "softplus":
        return nn.Softplus
    elif name == "mish":
        return nn.Mish
    elif name == "hardshrink":
        return nn.Hardshrink
    elif name == "none":
        return None


def train(model, optimizer, data, target, epochs, verbose=False):
    for epoch in range(epochs):
        if optimizer is not None:
            optimizer.zero_grad()

        predictions = model(data)
        loss = F.mse_loss(predictions, target.view(-1, 1))
        loss.backward()

        if optimizer is not None:
            optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            mae = torch.mean(torch.abs(data - predictions))
            print(f"Epoch: {epoch + 1}, MSE loss: {loss}, MAE: {mae}")


def test(model, data, target):
    with torch.no_grad():
        predictions = model(data)
        mae = torch.abs(data - predictions)
    return mae
