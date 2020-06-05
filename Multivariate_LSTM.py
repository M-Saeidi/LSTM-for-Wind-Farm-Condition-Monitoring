# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:28:31 2020

@author: Mostafa
"""
# Part 1 - Data Processing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import math
from datetime import datetime
from datetime import date

# Importing the dataset
dataset_P = scipy.io.loadmat('/content/drive/My Drive/LSTM_GPU/P.mat')
AP = np.array(dataset_P.get('P'))
P1 = np.array(AP[:,66])
dataset_Tgenc = scipy.io.loadmat('/content/drive/My Drive/LSTM_GPU/Tgencool.mat')
Tgc = np.array(dataset_Tgenc.get('Tgencool'))
Tgc1 = np.array(Tgc[:,66])
dataset_Tgen = scipy.io.loadmat('/content/drive/My Drive/LSTM_GPU/Tgen1.mat')
Tgen = np.array(dataset_Tgen.get('Tgen1'))
Tgen1 = np.array(Tgen[:,66])

# Filtering out the outliers
outliers1 = ((P1>2000) | (P1<(-20)))
outliers2 = ((Tgc1>180) | (Tgc1<(-20)))
outliers3 = ((Tgen1>180) | (Tgen1<(-20)))
Outliers = outliers1 | outliers2 | outliers3
Tgen1[Outliers] = None
P1[Outliers] = None
Tgc1[Outliers] = None

Nan_Values1 = np.isnan(P1)
Nan_Values2 = np.isnan(Tgc1)
Nan_Values3 = np.isnan(Tgen1)
Nan_Values = Nan_Values1 | Nan_Values2 | Nan_Values3
Tgen1[Nan_Values] = None
P1[Nan_Values] = None
Tgc1[Nan_Values] = None

Tgen1 = Tgen1[~np.isnan(Tgen1)]
P1 = P1[~np.isnan(P1)]
Tgc1 = Tgc1[~np.isnan(Tgc1)]


# Inputs:
Training_ratio = 0.75
Training_data = np.int(P1.shape[0]*Training_ratio)
Test_data = P1.shape[0]
Time_step = 12
Number_Features = 2


# Pre-processing the Inputs based on dropna
output = np.vstack(np.transpose([Tgen1]))
output = pd.DataFrame(output)
output_no_NAN = output.dropna()
output_processed = output_no_NAN

from pandas import DataFrame
from pandas import concat
def series_to_supervised(data, n_in=Time_step, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
            # put it all together
    agg = concat(cols, axis=1)
	# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Creating data structures
raw = DataFrame()
raw['ob1'] = P1[0:Training_data]
raw['ob2'] = Tgc1[0:Training_data]
values = raw.values
data_train = series_to_supervised(values)

raw = DataFrame()
raw['ob1'] = P1[Training_data:Test_data]
raw['ob2'] = Tgc1[Training_data:Test_data]
values = raw.values
data_test = series_to_supervised(values)

y_temp= []
for i in range(0, Training_data-Time_step+0):
    y_temp.append(output_processed.iloc[i, 0])
y_train = y_temp
y_train = np.array(y_train)
y_train = y_train.reshape(-1,1)
                          
y_temp= []
for i in range(Training_data+0, Test_data-Time_step+0):
    y_temp.append(output_processed.iloc[i, 0])
y_test = y_temp
y_test = np.array(y_test)
y_test = y_test.reshape(-1,1)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
data_1 = sc_X.fit_transform(data_train)
data_2 = sc_X.fit_transform(data_test)
sc_y_train = StandardScaler()
y_train = sc_y_train.fit_transform(y_train)
sc_y_test = StandardScaler()
y_test = sc_y_test.fit_transform(y_test)

X_train = np.reshape(data_1, (data_1.shape[0], Time_step, Number_Features))
X_test = np.reshape(data_2, (data_2.shape[0], Time_step, Number_Features))

############################################################################### Part 2 - Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 30, return_sequences = True, input_shape = (X_train.shape[1], 2)))
regressor.add(Dropout(0.1))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 30, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a third LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 30, return_sequences = True))
#regressor.add(Dropout(0.4))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 30, return_sequences = False))
regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32*8)

#########################################################################
predicted_LSTM = regressor.predict(X_test)
predicted_LSTM = sc_y_test.inverse_transform(predicted_LSTM)
y_test = sc_y_test.inverse_transform(y_test)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor_Linear = LinearRegression()
y_train = sc_y_train.inverse_transform(y_train)
regressor_Linear.fit(data_train, y_train)

# Predicting the Test set results
predicted_Linear = regressor_Linear.predict(data_test)

# Calculating the MSE
from sklearn.metrics import mean_squared_error 
MSE_LSTM_testing = mean_squared_error(y_test,predicted_LSTM)
MSE_Linear_testing = mean_squared_error(y_test,predicted_Linear)
from math import sqrt
RMS_LSTM_test = sqrt(MSE_LSTM_testing)
RMS_LSTM_test = sqrt(MSE_Linear_testing)
MSE_LSTM_training = mean_squared_error(y_train,sc_y_train.inverse_transform(regressor.predict(X_train))) 
MSE_Linear_training = mean_squared_error(y_train,regressor_Linear.predict(data_train)) 

print(MSE_LSTM_training)
print(MSE_LSTM_testing)
print(MSE_Linear_training)
print(MSE_Linear_testing)

Error_LSTM = y_test - predicted_LSTM
Error_Linear = y_test - predicted_Linear

import statsmodels.api as sm
c = sm.OLS(y_train,regressor_Linear.predict(data_train))
d = sm.OLS(y_test,sc_y_train.inverse_transform(regressor.predict(X_test)))
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
raw = DataFrame()
raw['ob1'] = P1
raw['ob2'] = Tgc1
values = raw.values
data_total = series_to_supervised(values)
data_3 = sc_X.fit_transform(data_total)
X_total = np.reshape(data_3, (data_3.shape[0], Time_step, Number_Features))
y_temp= []
for i in range(0,data_3.shape[0]):
    y_temp.append(output_processed.iloc[i, 0])
y_total = y_temp
y_total = np.array(y_total)
y_total = y_total.reshape(-1,1)

predicted_LSTM_total = regressor.predict(X_total)
predicted_LSTM_total = sc_y_train.inverse_transform(predicted_LSTM_total)
predicted_Linear_total = regressor_Linear.predict(data_total)
Error_LSTM_total = y_total - predicted_LSTM_total
Error_Linear_total = y_total - predicted_Linear_total


#[$^\circ$C]
Variable = P1
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
plt.plot(Variable, 'b.', label = 'Predicted LSTM Winding Temperature [C]')

plt.title('Time trace of generator active power WT 067', fontweight="bold", fontsize= 14)
plt.xlabel('Samples', fontweight="bold", fontsize= 18)
plt.ylabel('Generator Active Power [kW]', fontweight="bold", fontsize= 18)
plt.xlim(0, 100000)
plt.ylim(-2, 1800)
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
#ax.tick_params(axis='x', labelsize= 18)
#ax.tick_params(axis='y', labelsize= 18)
#plt.legend()
plt.grid(True)
plt.savefig("T7test.png")
plt.show()




from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
plt.plot(predicted_LSTM_total, 'b.', label = 'Predicted LSTM Winding Temperature [$^\circ$C]')
#plt.plot(Error_LSTM_total, 'r.', label = 'LSTM Predicted Error [$^\circ$C]')
plt.plot(Tgen1, 'r', label = 'Actual Winding Temperature [$^\circ$C]')
plt.title('Predicted and Actual Generator Winding Temperature WT 067', fontweight="bold", fontsize= 14)
plt.xlabel('Samples', fontweight="bold", fontsize= 18)
plt.ylabel('Generator Winding Temperature [$^\circ$C]', fontweight="bold", fontsize= 18)
plt.xlim(0, 100000)
plt.ylim(-20, 150)
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
#ax.tick_params(axis='x', labelsize= 18)
#ax.tick_params(axis='y', labelsize= 18)
plt.legend()
plt.grid(True)
plt.savefig("Predicted7test.png")
plt.show()
