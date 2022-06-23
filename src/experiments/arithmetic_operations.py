import numpy as np

import torch
import torch.nn.functional as F

from src.models.utils import ARITHMETIC_OPERATIONS
from src.models.mlp import MLP
from src.models.nac import NAC
from src.models.nalu import NALU

import src.models.utils as utils

from tqdm import tqdm

# Generate auxiliary constants
V_SIZE = 100
M, N, P, Q = 0, 10, 60, 100

# Model training constants
LR = 1e-2
EPOCHS = 10000

# Auxiliary constants
TABLE_PATH = "../../report/tables/"


def build_models():
    relu6 = MLP(input_dim=2, output_dim=1, num_hidden_layers=1, hidden_dim=2, activation="relu6")
    none = MLP(input_dim=2, output_dim=1, num_hidden_layers=1, hidden_dim=2, activation="none")
    nac = NAC(input_dim=2, output_dim=1, num_hidden_layers=1, hidden_dim=2)
    nalu = NALU(input_dim=2, output_dim=1, num_hidden_layers=1, hidden_dim=2)

    models = [relu6, none, nac, nalu]
    return models


def generate_sample(operator, right_bound, size):
    x = np.random.uniform(0, right_bound, size=(size, V_SIZE))
    a = np.sum(x[:, M:N], axis=1)
    b = np.sum(x[:, P:Q], axis=1)
    y = ARITHMETIC_OPERATIONS[operator](a, b)

    return a, b, y


def generate_test_data(operator, right_bound, test_size, max_values, interpolation):
    a_max, b_max, y_max = max_values
    x_test, y_test = [], []

    test_len = 0
    while test_len < test_size:
        a, b, y = generate_sample(operator, right_bound, test_size)
        i_range = (a < a_max) & (b < b_max) & (y < y_max)

        if interpolation:
            indices = i_range
        else:
            indices = ~i_range

        x_test.append(np.array([(a_, b_) for a_, b_ in zip(a, b)])[indices])
        y_test.append(y[indices])

        test_len += np.sum(indices)

    x_test = np.concatenate(x_test)[:test_size].astype(np.float32)
    y_test = np.concatenate(y_test)[:test_size].astype(np.float32)

    return x_test, y_test


def random_baseline(x_test_i, y_test_i, x_test_e, y_test_e, n_repeat=10):
    total_rmse_i, total_rmse_e = 0, 0

    for i in range(n_repeat):
        neural_net = MLP(input_dim=2, output_dim=1, num_hidden_layers=1, hidden_dim=2, activation="relu")
        X_i = torch.FloatTensor(x_test_i)
        X_e = torch.FloatTensor(x_test_e)
        y_i = torch.FloatTensor(y_test_i).view(-1, 1)
        y_e = torch.FloatTensor(y_test_e).view(-1, 1)
        with torch.no_grad():
            predictions_i = neural_net(X_i)
            rmse_i = torch.sqrt(F.mse_loss(predictions_i, y_i))

            predictions_e = neural_net(X_e)
            rmse_e = torch.sqrt(F.mse_loss(predictions_e, y_e))

        total_rmse_i += rmse_i
        total_rmse_e += rmse_e

    total_rmse_i /= n_repeat
    total_rmse_e /= n_repeat

    return total_rmse_i, total_rmse_e


def generate_data(operator, right_bound, train_size, test_size):
    a, b, y_train = generate_sample(operator, right_bound, train_size)

    max_values = [np.max(a), np.max(b), np.max(y_train)]

    x_test_i, y_test_i = generate_test_data(operator, right_bound, test_size, max_values, True)
    x_test_e, y_test_e = generate_test_data(operator, right_bound * 5, test_size, max_values, False)

    random_rmse_i, random_rmse_e = random_baseline(x_test_i, y_test_i, x_test_e, y_test_e)
    X_train = np.array([(a_, b_) for a_, b_ in zip(a, b)])

    data = [X_train, y_train, x_test_i, y_test_i, x_test_e, y_test_e, random_rmse_i, random_rmse_e]
    return data


def test(model, data, target):
    with torch.no_grad():
        predictions = model(data)
        rmse = torch.sqrt(F.mse_loss(predictions, target.view(-1, 1)))
    return rmse


def evaluate(X_train, y_train, X_test, y_test, r, results, op):
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    models = build_models()
    for model in models:
        print(f"Training {str(model)}")
        optimizer = torch.optim.Adam(model.parameters(), LR)
        utils.train(model, optimizer, X_train, y_train, EPOCHS)
        rmse = test(model, X_test, y_test).item()
        results[op].append(rmse / r * 100)


def save_results(results, path):
    with open(path, "w") as file:
        file.write("Relu6\tNone\tNAC\tNALU\n")
        for key, _ in results.items():
            file.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(*results[key]))


if __name__ == "__main__":
    results_i, results_e = {}, {}

    for op in tqdm(ARITHMETIC_OPERATIONS.keys()):
        print(f"\nTesting operation: {op}")
        results_i[op], results_e[op] = [], []

        X_train, y_train, X_test_i, y_test_i, X_test_e, y_test_e, r_i, r_e = generate_data(op, 1, 50000, 5000)
        evaluate(X_train, y_train, X_test_i, y_test_i, r_i, results_i, op)
        evaluate(X_train, y_train, X_test_e, y_test_e, r_e, results_e, op)

    save_results(results_i, TABLE_PATH + "interpolation.txt")
    save_results(results_e, TABLE_PATH + "extrapolation.txt")