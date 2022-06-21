'''
Main functions for calling stat methods, comparing with pre-existing packages, loading data etc
'''

import pandas as pd

data_path = f'E:\Documents\Datasets\Kaggle\House Prices\\train.csv'

if __name__ == '__main__':

    data = pd.read_csv(data_path)

    len(data)