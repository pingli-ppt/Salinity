# -*- coding: utf-8 -*-
import numpy as np

__all__ = ["mae", "mse", "rmse", "mape", "smape", "mase", 'wape', 'msmape',
           "mae_norm", "mse_norm", "rmse_norm", "mape_norm", "smape_norm",
           "mase_norm", 'wape_norm', 'msmape_norm']

# ====================== 工具函数 ======================
def _error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Simple error """
    actual = actual[..., 1]
    predicted = predicted[..., 1]
    return actual - predicted

def _percentage_error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Percentage error """
    actual = actual[..., 1]
    predicted = predicted[..., 1]
    eps = 1e-8
    return (actual - predicted) / (actual + eps)

# 适配 scaler 形状（修复 reshape 报错）
def _reshape(x):
    return x.reshape(-1, 1) if x.ndim == 1 else x

# ====================== 基础指标 ======================
def mse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.mean(np.square(_error(actual, predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.sqrt(mse(actual, predicted))

def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.mean(np.abs(_error(actual, predicted)))

def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    hist_data: np.ndarray,
    seasonality: int = 24,
    **kwargs
):
    actual = actual[..., 1]
    predicted = predicted[..., 1]
    hist_data = hist_data[..., 1]

    # 修复：去掉 seasonality==2 返回 -1
    n_hist = len(hist_data)
    if n_hist <= seasonality:
        return np.mean(np.abs(actual - predicted))
    
    #  naive 基准误差
    naive_mae = np.mean(np.abs(hist_data[seasonality:] - hist_data[:-seasonality]))
    if naive_mae < 1e-8:
        return np.mean(np.abs(actual - predicted))
    
    return np.mean(np.abs(actual - predicted)) / naive_mae

def mape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.mean(np.abs(_percentage_error(actual, predicted))) * 100

def smape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    actual = actual[..., 1]
    predicted = predicted[..., 1]
    eps = 1e-8
    return np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + eps)) * 100

def wape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    actual = actual[..., 1]
    predicted = predicted[..., 1]
    eps = 1e-8
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual)) + eps) * 100

def msmape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 0.1, **kwargs):
    actual = actual[..., 1]
    predicted = predicted[..., 1]
    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    return np.mean(2 * np.abs(predicted - actual) / denom) * 100

# ====================== 归一化指标 ======================
def _error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    actual = actual[..., 1]
    predicted = predicted[..., 1]

    actual = _reshape(actual)
    predicted = _reshape(predicted)

    a = scaler.inverse_transform(actual)
    p = scaler.inverse_transform(predicted)
    return a - p

def mse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return np.mean(np.square(_error_norm(actual, predicted, scaler)))

def rmse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return np.sqrt(mse_norm(actual, predicted, scaler))

def mae_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return np.mean(np.abs(_error_norm(actual, predicted, scaler)))

def mase_norm(
    actual: np.ndarray,
    predicted: np.ndarray,
    scaler: object,
    hist_data: np.ndarray,
    seasonality: int = 24,
    **kwargs
):
    actual = actual[..., 1]
    predicted = predicted[..., 1]
    hist_data = hist_data[..., 1]

    actual = _reshape(actual)
    predicted = _reshape(predicted)
    hist_data = _reshape(hist_data)

    a = scaler.inverse_transform(actual).ravel()
    p = scaler.inverse_transform(predicted).ravel()
    h = scaler.inverse_transform(hist_data).ravel()

    n_hist = len(h)
    if n_hist <= seasonality:
        return np.mean(np.abs(a - p))
    
    naive_mae = np.mean(np.abs(h[seasonality:] - h[:-seasonality]))
    if naive_mae < 1e-8:
        return np.mean(np.abs(a - p))
    
    return np.mean(np.abs(a - p)) / naive_mae

def mape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    err = _error_norm(actual, predicted, scaler)
    actual = actual[..., 1]
    actual = _reshape(actual)
    a = scaler.inverse_transform(actual).ravel()
    eps = 1e-8
    return np.mean(np.abs(err) / (np.abs(a) + eps)) * 100

def smape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    actual = actual[..., 1]
    predicted = predicted[..., 1]

    a = scaler.inverse_transform(_reshape(actual)).ravel()
    p = scaler.inverse_transform(_reshape(predicted)).ravel()
    eps = 1e-8
    return np.mean(2 * np.abs(a - p) / (np.abs(a) + np.abs(p) + eps)) * 100

def wape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    actual = actual[..., 1]
    predicted = predicted[..., 1]

    a = scaler.inverse_transform(_reshape(actual)).ravel()
    p = scaler.inverse_transform(_reshape(predicted)).ravel()
    eps = 1e-8
    return np.sum(np.abs(a - p)) / (np.sum(np.abs(a)) + eps) * 100

def msmape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, epsilon: float = 0.1, **kwargs):
    actual = actual[..., 1]
    predicted = predicted[..., 1]

    a = scaler.inverse_transform(_reshape(actual)).ravel()
    p = scaler.inverse_transform(_reshape(predicted)).ravel()
    comparator = np.full_like(a, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(p) + np.abs(a) + epsilon)
    return np.mean(2 * np.abs(p - a) / denom) * 100