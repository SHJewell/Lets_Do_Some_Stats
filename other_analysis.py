'''
This houses other analysis functions:

Index:
    VIF
'''

import sklearn.metrics as metrics
import regression as reg

def gen_vif(dset, primary):
    '''
    Calculates VIF values for all columns
    Calculation of the Variance Inflation Factor
    based on: https://online.stat.psu.edu/stat462/node/180/

    VIFj = 1 / (1 - R^2j)
    :return:
    '''

    cols = list(dset.columns).remove(primary)
    vif_results = dict()

    for col in cols:

        beta, alpha, y_pred = reg.skl_linear_reg(dset[primary], dset[col], do_pred=True)

        r2 = metrics.r2_score(dset[primary], y_pred)

        vif_results[col] = 1 / (1 - r2)

