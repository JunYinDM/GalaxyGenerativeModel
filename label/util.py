from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics


def r2(pred, target):
    return sklearn.metrics.r2_score(target, pred)


def mse(pred, target):
    return np.square(pred-target).mean()


def rmse(pred, target):
    return np.sqrt(mse(pred, target))


def mae(pred, target):
    return np.abs(pred-target).mean()


def pp_mse(pred, target):
    return np.square(pred-target).mean(axis=0)


def pp_rmse(pred, target):
    return np.sqrt(pp_mse(pred, target))


def pp_mae(pred, target):
    return np.abs(pred-target).mean(axis=0)
