'''
Errors

residuals
residual sum of squares
total sum of squares
rmse
R^2

'''

import numpy as np

def residuals(y_est, y_truth):

    return y_truth - y_est

def rmse(y_est, y_truth):

    return np.sum(np.sqrt((residuals(y_est, y_truth)) ** 2) / len(y_truth))

def rsquared(y_set, y_truth):
    '''
    R^2 Calculation

    R^2 = 1 - ( sum of squares / sum of residuals)
    '''

    sos = np.sum([(yn**2) for yn in y_set])
    sor = np.sum(y_truth - y_set)

    R2 = 1 - (sos /  sor)

    return R2

