'''
This houses other analysis functions:

Index:
    VIF
'''

import sklearn.metrics as metrics
import numpy as np

import regression as reg

def correlation_matrix(data, cols):
    '''

    :param data: data array, numpy array or pandas DataFrame
    :param cols: list of column names
    :return:
    '''


def gen_vif(dset, primary):
    '''
    Calculates VIF values for all columns
    Calculation of the Variance Inflation Factor
    based on: https://online.stat.psu.edu/stat462/node/180/

    VIFj = 1 / (1 - R^2j)
    :return:
    '''

    cols = list(dset.columns)
    vif_results = dict()

    dset_np = dset[primary].to_numpy()

    for col in cols:

        if col == primary:
            continue

        temp = dict()
        y = dset[col].to_numpy()

        beta, alpha, y_pred = reg.skl_linear_reg(dset_np, y, pred=True)

        temp['R2'] = metrics.r2_score(dset[primary], y_pred)

        if temp['R2'] == 1:
            temp['VIF'] = np.NINF
        else:
            temp['VIF'] = 1 / (1 - temp['R2'])

        vif_results[col] = temp

    return vif_results

def tss(y):
    '''
    Total sum of squares
    '''

    return np.sum((y - np.mean(y))**2)


def rss(y, fy):
    '''
    Residual Sum of Squares
    :return:
    '''

    return np.sum((y - fy) ** 2)