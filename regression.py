'''
Starting review of stats and learning datascience techniques, let's start at the beginning with regressions

Methods:
    Linear Regression
    Nearest Neighbor
    Lasso Regression
    Multivariate Regression
'''

import numpy as np
from sklearn import linear_model

def my_least_squares_reg(x, y):
    '''
    Least squares linear regression. Looks like it will be computationally costly

    RSS(beta)  = sum( (y = xT * beta)^2 )
    beta = (XT * X)^(-1) * XT * y
        if XT * X is non-singular

    :param x:       numpy array
    :param y:       numpy array
    :return beta:   numpy array
    '''

    y_mean = np.mean(y)
    x_mean = np.mean(x)

    beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    alpha = y_mean - (beta * x_mean)

    # beta = (1 / (np.transpose(x) * x)) * np.transpose(x) * y
    # rss = np.sum(y - beta*x)
    #
    # alpha = np.mean((y - (rss * x)))

    return beta, alpha

def my_linear_reg(x, y):
    '''
    Simplest rise-over-run regression

    beta = mean(y) / mean(x)
    beta = mean(y - beta * x)

    :param x:
    :param y:
    :return beta, alpha:    numpy array, scalar float
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

