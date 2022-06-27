'''
Main functions for calling stat methods, comparing with pre-existing packages, loading data etc

housing data from:

'''

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import scipy
from matplotlib import pyplot as plt

import regression
import error_calcs
import plotters

import regression_tests as regtest

# house_data_path = f'E:\Documents\Datasets\Kaggle\House Prices\\train.csv'
house_data_path = 'C:\\Users\\Scrooge\\Documents\\Data\\Kaggle\\Housing\\train.csv'


if __name__ == '__main__':

    data = pd.read_csv(house_data_path)

    for col in data.columns:

        # if col == 'SalePrice':
        #
        #     continue

        if is_numeric_dtype(data[col]):

            selected_data = data[['SalePrice', col]]
            selected_data.dropna(axis='index', inplace=True)

            regtest.linear_reg_test(selected_data, col, 'SalePrice')