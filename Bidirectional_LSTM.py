# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:32:02 2020

@author: Mostafa
"""
# Part 1 - Data Processing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io

# Importing the dataset
dataset_Tgen = scipy.io.loadmat('Tgen1.mat')
Tgen = np.array(dataset_Tgen.get('Tgen1'))
Tgen1 = np.array(Tgen[:,6])

# Filtering out the outliers
outliers1 = ((Tgen1>180) | (Tgen1<(-20)))
Outliers = outliers1
Tgen1[Outliers] = None

Nan_Values1 = np.isnan(Tgen1)
Nan_Values = Nan_Values1
Tgen1[Nan_Values] = None

Tgen1 = Tgen1[~np.isnan(Tgen1)]
Tgen1 = Tgen1.reshape(-1,1)

Training_data = np.int(Tgen1.shape[0]*0.75)
Test_data = Tgen1.shape[0]
Time_step = 12


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Dataset_scaled = sc.fit_transform(Tgen1)


# Creating a data structures with 12 timesteps and 1 output
X_train = []
y_train = []
for i in range(Time_step, Training_data):
    X_train.append(Dataset_scaled[i-Time_step:i, 0])
    y_train.append(Dataset_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

############################################################################### Part 2 - Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(Bidirectional(LSTM(units = 30, return_sequences = True, input_shape = (X_train.shape[1], 1))))
regressor.add(Dropout(0.1))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(Bidirectional(LSTM(units = 30, return_sequences = True)))
regressor.add(Dropout(0.1))

# Adding a third LSTM layer and some Dropout regularisation
#regressor.add(Bidirectional(LSTM(units = 50, return_sequences = True)))
#regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(Bidirectional(LSTM(units = 30, return_sequences = False)))
regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32*8)

############################################################################### Part 3  - Making the predictions and visualising the results

X_test = []
y_test = []
for i in range(Training_data,Test_data):
    X_test.append(Dataset_scaled[i-Time_step:i, 0])
    y_test.append(Dataset_scaled[i, 0])
y_test = np.array(y_test)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted_winding_temperature = regressor.predict(X_test)

# Scaling Back the Features and Outputs
y_train = y_train.reshape(-1,1)
y_train = sc.inverse_transform(y_train)
y_test = y_test.reshape(-1,1)
y_test = sc.inverse_transform(y_test)
predicted_winding_temperature = sc.inverse_transform(predicted_winding_temperature)

# Calculating the MSE
from sklearn.metrics import mean_squared_error 
from math import sqrt
MSE_LSTM_Regression_training = mean_squared_error(y_train,sc.inverse_transform(regressor.predict(X_train))) 
MSE_LSTM_Regression_testing = mean_squared_error(y_test,predicted_winding_temperature)
RMSE_LSTM_train= sqrt(MSE_LSTM_Regression_training)
RMSE_LSTM_test = sqrt(MSE_LSTM_Regression_testing)

print(MSE_LSTM_Regression_training)
print(MSE_LSTM_Regression_testing)

Error_LSTM = y_test - predicted_winding_temperature

import statsmodels.api as sm
c = sm.OLS(y_train,sc.inverse_transform(regressor.predict(X_train)))
d = sm.OLS(y_test,predicted_winding_temperature)
c2 = c.fit()
d2 = d.fit()
print("summary()\n",c2.summary())
print("pvalues\n",c2.pvalues)
print("tvalues\n",c2.tvalues)
print("rsquared\n",c2.rsquared)
print("rsquared_adj\n",c2.rsquared_adj)
print("summary()\n",d2.summary())
print("pvalues\n",d2.pvalues)
print("tvalues\n",d2.tvalues)
print("rsquared\n",d2.rsquared)
print("rsquared_adj\n",d2.rsquared_adj)



############################################### Total Data
X_total = []
y_total = []
for i in range(Time_step, Test_data):
    X_total.append(Dataset_scaled[i-Time_step:i, 0])
    y_total.append(Dataset_scaled[i, 0])
X_total, y_total = np.array(X_total), np.array(y_total)
X_total = np.reshape(X_total, (X_total.shape[0], X_total.shape[1], 1))

predicted_LSTM_total = regressor.predict(X_total)
predicted_LSTM_total = sc.inverse_transform(predicted_LSTM_total)
Error_LSTM_total = y_total - predicted_LSTM_total

from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
plt.plot(Error_LSTM_total, 'r', label = 'LSTM Predicted Error [C]')
plt.plot(predicted_LSTM_total, 'b', label = 'Predicted LSTM Winding Temperature [C]')
#plt.plot(predicted_Linear_total, 'r', label = 'Predicted Linear Winding Temperature [C]')

plt.title('Actual and Predicted Generator Winding Temperature WT 007', fontweight="bold")
plt.xlabel('Samples')
plt.ylabel('Winding Temperature [C]')
plt.xlim(0, 100000)
plt.ylim(-20, 150)
plt.legend()
plt.grid(True)
plt.savefig("Signals67.png")
plt.show()