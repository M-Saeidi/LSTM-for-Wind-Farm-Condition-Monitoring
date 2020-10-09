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
