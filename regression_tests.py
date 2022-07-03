import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import regression
import error_calcs


def linear_reg_test(data_path):

    data = pd.read_csv(data_path)

    yearbuilt = data['YearBuilt'].to_numpy()
    price = data['SalePrice'].to_numpy()

    #plts.scatter_2d(price, yearbuilt, y_label='Year Built', x_label='Price')

    my_alpha, my_intercept = regression.my_linear_reg(yearbuilt, price)
    sk_alpha, sk_intercept, sk2_model_price = regression.skl_linear_reg(yearbuilt, price, pred=True)

    my_model_price = yearbuilt * my_alpha + my_intercept
    sk1_model_price = yearbuilt * sk_alpha[0] + sk_intercept

    my_residual = np.absolute(price - my_model_price)
    sk1_residual = np.absolute(price - sk1_model_price)
    sk2_residual = np.absolute(price - sk2_model_price)

    model_resid_1 = np.absolute(my_model_price - sk1_model_price)
    model_resid_2 = np.absolute(my_model_price - sk2_model_price)

    print(error_calcs.rmse(my_model_price, price), error_calcs.rmse(sk1_model_price, price), error_calcs.rmse(sk2_model_price, price))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.scatter(yearbuilt, price, marker='.', color='blue', label='Price')
    ax1.scatter(yearbuilt, my_model_price, marker='+', color='green', label='My Price')
    ax1.scatter(yearbuilt, sk1_model_price, marker='*', color='red', label='SKL1 Price')
    ax1.scatter(yearbuilt, sk2_model_price, marker='*', color='firebrick', label='SKL2 Price')
    ax1.xtitle='Price'
    ax1.legend()
    ax2.scatter(yearbuilt, my_residual, marker='+', color='green', label='My Resid')
    ax2.scatter(yearbuilt, sk1_residual, marker='*', color='red', label='SKL1 Resid')
    ax2.scatter(yearbuilt, sk2_residual, marker='*', color='firebrick', label='SKL2 Resid')
    ax2.legend()
    ax3.scatter(yearbuilt, model_resid_1, marker='.', label='Model Resid 1')
    ax3.scatter(yearbuilt, model_resid_2, marker='.', label='Model Resid 2')
    ax3.legend()

    plt.show()