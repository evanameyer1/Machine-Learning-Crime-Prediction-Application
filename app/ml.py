import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima import ADFTest
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose 

plt.style.use('dark_background')

# load the dataset
df = pd.read_csv('../scoring_datasets/final_aggregation.csv')

df['date'] = pd.to_datetime(df['date'])

#Is the data stationary?
adf_test = ADFTest(alpha = 0.05)
adf_test.should_diff(df)
#Not stationary...

#Dickey-Fuller test

adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df)
print("pvalue = ", pvalue, " if above 0.05, data is not stationary")
#Since data is not stationary, we may need SARIMA and not just ARIMA