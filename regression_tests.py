import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import regression
import error_calcs


def linear_reg_test(data, x_name, y_name):

    yearbuilt = data[x_name].to_numpy()
    price = data[y_name].to_numpy()

    #plts.scatter_2d(price, yearbuilt, y_label='Year Built', x_label='Price')

    my_alpha, my_intercept = regression.my_linear_reg(yearbuilt, price)
    sk_alpha, sk_intercept, sk_model_price = regression.skl_linear_reg(yearbuilt, price, pred=True)

    my_model_price = yearbuilt * my_alpha + my_intercept

    my_residual = np.absolute(price - my_model_price)
    sk_residual = np.absolute(price - sk_model_price)

    model_resid = np.absolute(my_model_price - sk_model_price)

    print(error_calcs.rmse(my_model_price, price), error_calcs.rmse(sk_model_price, price))

    plt.subplot(3, 1, 1)
    plt.scatter(yearbuilt, price, marker='.', color='blue', label='Price')
    plt.plot(yearbuilt, my_model_price, color='green', label='My Price')
    plt.plot(yearbuilt, sk_model_price, color='red', label='SKL1 Price')
    plt.title(f'{x_name} vs {y_name}')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.scatter(yearbuilt, my_residual, marker='+', color='green', label='My Resid')
    plt.scatter(yearbuilt, sk_residual, marker='*', color='red', label='SKL1 Resid')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.scatter(yearbuilt, model_resid, marker='.', label='Model Resid')
    plt.legend()

    plt.show()