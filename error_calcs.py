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

    return np.sqrt((residuals(y_est, y_truth)) ** 2 / len(y_truth))


