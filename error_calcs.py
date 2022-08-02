'''
Errors

residuals
residual sum of squares
total sum of squares
rmse
R^2

'''

import numpy as np
from numpy.linalg import norm

def residuals(y_est, y_truth):

    return y_truth - y_est

def rmse(y_est, y_truth):

    return np.sum(np.sqrt((residuals(y_est, y_truth)) ** 2) / len(y_truth))

def rsquared(y_truth, y_set, **kwargs):
    '''
    R^2 Calculation

    R^2 = 1 - ( sum of squares / sum of residuals)
    '''

    normalize = kwargs.get('norm', False)

    if normalize:

        y_set = y_set / norm(y_set)
        y_truth = y_truth / norm(y_truth)

    sos = np.sum((y_truth - y_set)**2)
    sor = np.sum((y_set - np.mean(y_set))**2)

    R2 = 1 - (sos / sor)

    return R2

