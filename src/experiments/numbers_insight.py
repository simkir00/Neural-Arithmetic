import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

from src.models import utils as utils
from src.models.mlp import MLP

from tqdm import tqdm

# Auxiliary constants
IMG_PATH = "../../report/images/"

# Paper experiment hyperparameters
TRAIN_BORDER = [-5, 6]
TEST_BORDER = [-20, 21]

IN_DIMS = 1
OUT_DIMS = 1
N_HIDDEN_LAYERS = 3
HIDDEN_DIM = 8

N_INSTANCES = 100
EPOCHS = 10000
LR = 1e-2


def get_mae(train_data, test_data, verbose=False):
    result = []
    for activation_func in utils.ACTIVATION_FUNCTIONS:
        maes = []
        rrange = range(N_INSTANCES)
        if verbose:
            print(f"\nCurrent activation function: {activation_func}")
            rrange = tqdm(rrange)
        for i in rrange:
            current_mlp = MLP(input_dim=IN_DIMS, output_dim=OUT_DIMS,
                              num_hidden_layers=N_HIDDEN_LAYERS, hidden_dim=HIDDEN_DIM,
                              activation=activation_func)
            # optimizer = None
            optimizer = Adam(current_mlp.parameters(), LR)
            utils.train(current_mlp, optimizer, train_data, train_data, EPOCHS, verbose=False)
            test_mae = utils.test(current_mlp, test_data, test_data)
            maes.append(test_mae)

        mean_mae = torch.cat(maes, dim=1).mean(dim=1)
        result.append(mean_mae)

    result = [x.numpy().flatten() for x in result]
    return result


def plot(maes, borders):
    fig, ax = plt.subplots(figsize=(9, 8))
    x = np.arange(*borders)

    for mae, activation in zip(maes, utils.ACTIVATION_FUNCTIONS):
        # fig_tmp, ax_tmp = plt.subplots(figsize=(9, 8))
        # ax_tmp.plot(x, mae, label=activation)
        # plt.legend(loc="best")
        # plt.savefig(IMG_PATH + f"numbers_insight_{activation}.png", format="png", dpi=300)

        ax.plot(x, mae, label=activation)

    plt.ylabel("Mean Absolute Error")
    plt.legend(loc="best")
    plt.savefig(IMG_PATH + f"numbers_insight.png", format="png", dpi=300)
    plt.show()


if __name__ == "__main__":
    train_data = torch.arange(*TRAIN_BORDER).unsqueeze_(1).float()
    test_data = torch.arange(*TEST_BORDER).unsqueeze_(1).float()

    maes = get_mae(train_data, test_data, verbose=True)
    plot(maes, TEST_BORDER)

    # print(torch.cat([torch.Tensor([1, 2, 3]), torch.Tensor([3, 2, 1])], dim=0))
