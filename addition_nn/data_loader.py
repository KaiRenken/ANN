import torch


def get_dataset(size):
    X = torch.rand((size, 2)) * 100
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = x1 + x2
    return X, y
