# Stock Prediction

# In this project, I used a Recurrent Neural Network regressor with the Long Term Short Memory model and time series to predict the stock price of American Airlines (AAL) in 2020. 
# LTSM is used avoid technical problems with optimization of RNNs.

## 1. Load Data

# The input data is a time series data from 2013 to 2018 of different stock tickers. The data is initially normalized using a minmax normalizer.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# Importing the training set
dataset = pd.read_csv('./all_stocks_5yr.csv')
dataset_cl = dataset[dataset['Name']=='AAL'].close.values

# Normalization
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

dataset_cl = dataset_cl.reshape(dataset_cl.shape[0], 1)
dataset_cl = sc.fit_transform(dataset_cl)

## 2.  Cutting our time series into sequences

# The time series stock price data are split into sequnces (windows) 
# of length $L$, starting at time $t$, and end at time $t + L - 1$, 
# as inputs to the LSTM neural network. The outputs that the model is 
# trying to predict are the stock prices at time $t+L$.

# The model here uses a 5-day look back (L=5) for the model
def processData(data, lb):
    X, Y = [], []
    for i in range(len(data) - lb - 1):
        X.append(data[i: (i + lb), 0])
        Y.append(data[(i + lb), 0])
    return np.array(X), np.array(Y)
    
X, y = processData(dataset_cl, 5)