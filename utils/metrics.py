import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))

def nRMSE(pred, true, ymax=400):
    return np.sqrt(np.mean((true - pred) ** 2)) / ymax

def nMAE(pred, true, ymax=400):
    return np.mean(np.abs(true - pred)) / ymax

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    nrmse = nRMSE(pred, true, ymax=1366)
    nmae = nMAE(pred, true, ymax=1366)
    return mae, mse, nrmse, nmae

def metric_with_ymax(pred, true, ymax):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    nrmse = nRMSE(pred, true, ymax=ymax)
    nmae = nMAE(pred, true, ymax=ymax)
    return mae, mse, nrmse, nmae
