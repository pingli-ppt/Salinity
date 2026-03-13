# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.utils.timefeatures import time_features

def train_val_split(train_data, ratio, seq_len):
    if ratio == 1:
        return train_data, None

    elif seq_len is not None:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_before(train_data, border)
        train_data_rest, valid_data = split_before(train_data, border - seq_len)
        return train_data_value, valid_data
    else:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_before(train_data, border)
        return train_data_value, valid_data_rest

def forecasting_data_provider(data, config, timeenc, batch_size, shuffle, drop_last):
    dataset = DatasetForTransformer(
        dataset=data,
        history_len=config.seq_len,
        prediction_len=config.pred_len,
        label_len=config.label_len,
        timeenc=timeenc,
        freq=config.freq,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )

    return dataset, data_loader

def get_time_mark(
    time_stamp: np.ndarray,
    timeenc: int,
    freq: str,
) -> np.ndarray:
    """
    Extract temporal features from the time stamp.

    :param time_stamp: The time stamp ndarray.
    :param timeenc: The time encoding type.
    :param freq: The frequency of the time stamp.
    :return: The mark of the time stamp.
    """
    if timeenc == 0:
        origin_size = time_stamp.shape
        data_stamp = decompose_time(time_stamp.flatten(), freq)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    elif timeenc == 1:
        origin_size = time_stamp.shape
        data_stamp = time_features(pd.to_datetime(time_stamp.flatten()), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    else:
        raise ValueError("Unknown time encoding {}".format(timeenc))
    return data_stamp.astype(np.float32)