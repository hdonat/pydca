#!/usr/bin/env python3

"""
This module contains only metric and loss functions
"""

from sklearn.metrics import root_mean_squared_error


def calc_rmse_matrix_flatten(target, prediction):
    """Root mean squared error between two matrices
    :param target: target values
    :type target: numpy array
    :param prediction: prediction values
    :type prediction: numpy array
    :return: rmse
    :rtype: float
    """
    rmse = root_mean_squared_error(target, prediction, squared=False)
    return rmse
