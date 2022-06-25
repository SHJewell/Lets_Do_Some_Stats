'''
Main functions for calling stat methods, comparing with pre-existing packages, loading data etc
'''

import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt

import regression
import error_calcs
import plotters

data_path = f'E:\Documents\Datasets\Kaggle\House Prices\\train.csv'

if __name__ == '__main__':

    data = pd.read_csv(data_path)

    yearbuilt = data['YearBuilt'].to_numpy()
    price = data['SalePrice'].to_numpy()

    #plts.scatter_2d(price, yearbuilt, y_label='Year Built', x_label='Price')

    my_alpha, my_intercept = regression.my_linear_reg(price, yearbuilt)
    sk_alpha, sk_intercept = regression.skl_linear_reg(price, yearbuilt)

    my_model_price = yearbuilt * my_alpha + my_intercept
    sk_model_price = yearbuilt * sk_alpha[0] + sk_intercept

    my_residual = np.absolute(price - my_model_price)
    sk_residual = np.absolute(price - sk_model_price)

    print(error_calcs.rmse(my_model_price, price), error_calcs.rmse(sk_model_price, price))

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.scatter(yearbuilt, price, marker='.', color='blue', label='Price')
    ax1.scatter(yearbuilt, my_model_price, marker='+', color='green',label='My Price')
    ax1.scatter(yearbuilt, sk_model_price, marker='*', color='red', label='SKL Price')
    ax1.legend()
    ax2.scatter(yearbuilt, my_residual, marker='+', color='green', label='My Resid')
    ax2.scatter(yearbuilt, sk_residual, marker='*', color='red', label='SKL Resid')

    plt.show()
