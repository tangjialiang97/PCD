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

<<<<<<< HEAD
def _as_ymax_array(ymax, pred):
    if isinstance(ymax, (list, tuple, np.ndarray)):
        ymax_arr = np.asarray(ymax, dtype=np.float32)
        if pred.ndim >= 1 and ymax_arr.ndim == 1:
            # Match channel dimension for multivariate prediction
            if pred.shape[-1] != ymax_arr.shape[0]:
                raise ValueError(
                    f"ymax length ({ymax_arr.shape[0]}) does not match "
                    f"pred channels ({pred.shape[-1]})"
                )
            reshape_dims = (1,) * (pred.ndim - 1) + (ymax_arr.shape[0],)
            return ymax_arr.reshape(reshape_dims)
        return ymax_arr
    return float(ymax)


def nRMSE(pred, true, ymax=400):
    ymax_arr = _as_ymax_array(ymax, pred)
    return np.sqrt(np.mean((true - pred) ** 2)) / ymax_arr

def nMAE(pred, true, ymax=400):
    ymax_arr = _as_ymax_array(ymax, pred)
    return np.mean(np.abs(true - pred)) / ymax_arr

def metric(pred, true, ymax):
    # For multivariate predictions, compute per-channel metrics then average.
    if pred.ndim >= 2 and pred.shape[-1] > 1:
        axes = tuple(range(pred.ndim - 1))
        mae_per = np.mean(np.abs(true - pred), axis=axes)
        mse_per = np.mean((true - pred) ** 2, axis=axes)
        ymax_arr = _as_ymax_array(ymax, pred).reshape(-1)
        nrmse_per = np.sqrt(mse_per) / ymax_arr
        nmae_per = mae_per / ymax_arr
        mae = float(np.mean(mae_per))
        mse = float(np.mean(mse_per))
        nrmse = float(np.mean(nrmse_per))
        nmae = float(np.mean(nmae_per))
        return mae, mse, nrmse, nmae
    else:
        mae = MAE(pred, true)
        mse = MSE(pred, true)
        nrmse = nRMSE(pred, true, ymax=ymax)
        nmae = nMAE(pred, true, ymax=ymax)
        return mae, mse, nrmse, nmae
=======
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
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1
