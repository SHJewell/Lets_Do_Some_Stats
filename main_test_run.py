'''
Main functions for calling stat methods, comparing with pre-existing packages, loading data etc
'''

import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model
import statsmodels.api as sm

import regression
import error_calcs
import plotters
import other_analysis as other
import regression_tests as regtest

house_data_path = f'E:\\Documents\\Datasets\\Kaggle\\House Prices\\train.csv'
small_bp_set = f'E:\\Documents\\Datasets\\Toy Sets\\Bloodpress\\bloodpress.csv'
toy_stocks_set =f'E:\Documents\Datasets\Toy Sets\Stockmarket\stockmarket.csv'

if __name__ == '__main__':

    bp = pd.read_csv(small_bp_set)

    xs = ['Pt', 'Age', 'Weight', 'BSA', 'Dur', 'Pulse', 'Stress']

    bpy = bp['BP']


    bp_dy, bp_dX = dmatrices('BP ~ Age+Weight+BSA+Dur+Pulse+Stress', data=bp, return_type='dataframe')

    their_vif = pd.DataFrame()
    their_vif['VIF'] = [variance_inflation_factor(bp_dX.values, i) for i in range(bp_dX.shape[1])]
    their_vif['variable'] = bp_dX.columns

    my_vif = other.gen_vif(bp, 'BP')

    print(pd.DataFrame.from_dict(my_vif).transpose(), their_vif)
    #regtest.my_least_squares_reg(house_data_path))