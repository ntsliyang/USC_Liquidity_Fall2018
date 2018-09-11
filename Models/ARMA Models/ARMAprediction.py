import wrds
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA
from arch import arch_model

db = wrds.Connection()
# andrew95
# Cynfudan95

db.list_libraries()

data_merged = db.raw_sql("select permno, prc, shrout, date, hsiccd from crspq.dsf where prc>0 and shrout*prc < 50000 and hsiccd < 3600 and hsiccd >3500 and date = '2012-10-22'")

data_stock = db.raw_sql("select permno, vol, date from crspq.dsf where permno= 11394.0 and date > '2012-10-22'")

dataframe=data_stock[['date', 'vol']]

series=dataframe.set_index('date')


difference=series-series.shift()
difference.dropna(inplace=True)


split_point = round(len(difference) *0.8)
dataset, validation = difference[0:split_point], difference[split_point:]

plot_acf(dataset, lags=10)

plot_pacf(dataset, lags=10)

mod=ARMA(dataset, order=(7,2))
result=mod.fit()
print(result.summary())


forecast = result.forecast(steps=len(validation))[0]

error=np.transpose(validation)-forecast
err2=np.transpose(np.square(error))
RMSE=np.sqrt(err2.sum()/len(validation))