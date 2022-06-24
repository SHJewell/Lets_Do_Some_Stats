'''
Main functions for calling stat methods, comparing with pre-existing packages, loading data etc
'''

import pandas as pd

import regression as reg
import error_calcs as ec
import plotters as plts

data_path = f'E:\Documents\Datasets\Kaggle\House Prices\\train.csv'

if __name__ == '__main__':

    data = pd.read_csv(data_path)

    yearbuilt = data['YearBuilt'].to_numpy()
    price = data['SalePrice'].to_numpy()

    plts.scatter_2d(price, yearbuilt, y_label='Year Built', x_label='Price')

    print(reg.my_linear_reg(price, yearbuilt))
    print(reg.skl_linear_reg(price, yearbuilt))

    len(data)