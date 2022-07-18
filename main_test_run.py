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
import other_analysis as otal
import regression_tests as regtest

house_data_path = f'E:\Documents\Datasets\Kaggle\House Prices\\train.csv'
small_bp_set = f'E:\Documents\Datasets\General\Bloodpress\\bloodpress.csv'

if __name__ == '__main__':

    data = pd.read_csv(small_bp_set)

    regtest.my_least_squares_reg(house_data_path)