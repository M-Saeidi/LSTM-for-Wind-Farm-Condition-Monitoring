# LSTM for Wind Farm Condition Monitoring
Deep learning application for fault detection and condition monitoring in wind farms
large portion of costs of a wind farm is related to condition monitoring and O&M. Among different existing strategies for mitigation of O&M costs, the methods which use SCADA-based datasets are more popular since these systems are already installed which can be interpreted as decrease in expenses.

In the recent decade, ML algorithms emerged as one of the most potent and proficient tools to overcome many industrial challenges. Hence, many DL models are developed in the literature to make condition monitoring in wind farms easier and more accurate. In this project, the main established methods for condition monitoring in wind farms are investigated and RNN is set as the focal point of interest due to their considerable capabilities for handling time-series data.

In this project, several preprocessing techniques for missing value and outliers and scaling techniques are exploited to prepare the data for ML models. As the first model, a simple linear regression model is developed to predict the winding temperature of generators. Moreover, correlation analysis is used to reduce the dimensionality of inputs features for Multivariate modeling from more than 35 features to 2 variables. Furthermore, different types of LSTM such as Stacked LSTM, Bidirectional LSTM, and Multivariate LSTM are developed to predict temperature with higher precision. To demonstrate the effectiveness of LSTM models, these models are evaluated based on different criteria such as MSE and R-squared. For this purpose, two case scenarios are considered based on the performance of a wind farm in the North of Quebec Province, Canada in collaboration with Power Factors.

# Initialization
The data is available in the LSTM-for-Wind-Farm-Condition-Monitoring folder. These data are provided by Power Factors which is one of the prominent companies in providing condition monitoring and asset management for renewable energy sector. I will start with importing the libraries.
```python
# Initialization
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import math
from datetime import datetime
from datetime import date
%tensorflow_version 1.x
```

Lets import the datasets and take a look at a small slice of our data to see what features and data types we have.
```python
# Importing the dataset
dataset_P = scipy.io.loadmat('.../My Drive/LSTM_GPU/P.mat')
AP = np.array(dataset_P.get('P'))
P1 = np.array(AP[:,66])
dataset_Tgenc = scipy.io.loadmat('.../My Drive/LSTM_GPU/Tgencool.mat')
Tgc = np.array(dataset_Tgenc.get('Tgencool'))
Tgc1 = np.array(Tgc[:,66])
dataset_Tgen = scipy.io.loadmat('.../My Drive/LSTM_GPU/Tgen1.mat')
Tgen = np.array(dataset_Tgen.get('Tgen1'))
Tgen1 = np.array(Tgen[:,66])
```

```python
Temp = {'Active Power':P1,
       'Cooling System Temperature':Tgc1,
       'Generator Temperature':Tgen1}
Temp1 = pd.DataFrame(Temp)
Temp1.head()
```

|   | Active Power  | Cooling System Temperature | Generator Temperature  |
| - | :---------:   |  :---------:               |   :--------:           |
| 0 | 156.740829    |  39.0                      | 49.426979              |
| 1 | 309.560822    |  39.0                      | 49.305450              |
| 2 | 504.298340    |  39.0                      | 49.183918              |
| 3 | 227.077560    |  39.0                      | 49.062386              |
| 4 | -5.392981     |  39.0                      | 46.403320              |

Active Power: Produced active power [kW] \n
Cooling System Temperature: Temperature of generator cooling system [°C] \n
Generator Temperature: Temperature of generator windings [°C] \n

# Data Preprocessing & Cleaning
The datasets have many outliers which can be removed by imposing upper and lower limits based on the engineering knowledge for each feature.
```python
# Filtering out the outliers
outliers1 = ((P1>2000) | (P1<(-20)))
outliers2 = ((Tgc1>180) | (Tgc1<(-20)))
outliers3 = ((Tgen1>180) | (Tgen1<(-20)))
Outliers = outliers1 | outliers2 | outliers3
Tgen1[Outliers] = None
P1[Outliers] = None
Tgc1[Outliers] = None
```
Moreover, the SCADA systems which are responsible for gathering data have failed to measure the features in some period of times. The failure of measuring systems produce many NaN values in our dataset which should be handled.
```python
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
```

# Data Preparation for LSTM Model
Before preparing datasets for LSTM moel, the training set ratio, LSTM time_step, and number of features are defined.
```python
# Inputs:
Training_ratio = 0.75
Training_data = np.int(P1.shape[0]*Training_ratio)
Test_data = P1.shape[0]
Time_step = 12
Number_Features = 2
```
Since the data collection frequency is 10-min, the time-step is chosen be to 12. In the other word, 12*10 = 120 mins (2Hrs) is sufficient for considering the temperature variation of generator windings. Hence, the datasets are converted to suitable format for LSTM model which includes the previous samples (based on the defined time-steps).
```python
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
```
Since the datasets are preprocessed and converted to the proper structure for LSTM model, they can be scaled using StandardScaler library to obtain accurate results.
```python
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
```
Now, the datasets are ready to be fed to our Deep Learning model. Long short-term memory (LSTM) models are part of Recurrent Neural Networks (RNNs) which show excellent performance for time-series and sequential problems. Since this problem is also a sequential anomaly detection project based on the 10-min average SCADA datasets, several LSTM models will be developed to predict the windings temperature of generator.
```python
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
```


