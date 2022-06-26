'''
Starting review of stats and learning datascience techniques, let's start at the beginning with regressions

Methods:
    Linear Regression
    Lasso Regression
    Multivariate Regression
'''

import numpy as np
from sklearn import linear_model

def my_linear_reg(x, y):
    '''
    Simplest rise-over-run regression
    :param x:
    :param y:
    :return:
    '''

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    beta = y_mean / x_mean
    alpha = np.mean((y - (beta * x)))

    return beta, alpha

def skl_linear_reg(x, y, **kwargs):

    do_pred = kwargs.get('pred', False)

    x = x.reshape(-1, 1)

    # getting weird NoneType error here. Unsure why
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    if do_pred:

        y_pred = regr.predict(x)

        return regr.coef_, regr.intercept_, y_pred

    return regr.coef_, regr.intercept_

