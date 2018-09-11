import wrds
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
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

series.plot()
plt.show()

difference=series-series.shift()
difference.dropna(inplace=True)
print(difference)

difference.plot()
plt.show()
plot_acf(difference, lags=10)
plt.show()

plot_pacf(difference, lags=10)
plt.show()

mod=ARMA(difference, order=(7,2))
result=mod.fit()
print(result.summary())


wnoise=result.resid
wnoise2=np.square(wnoise)

plot_acf(wnoise2, lags=10)

plt.show()
plot_pacf(wnoise2, lags=10)
plt.show()

modwnoise = arch_model(difference,vol='EGARCH',p=2,o=0,q=1)
egarchresult=modwnoise.fit()
print(egarchresult.summary())

diff2=difference.multiply(0.01)
modwnoise2 = arch_model(diff2,vol='EGARCH',p=2,o=0,q=1, dist='studentst')
egarchresult2=modwnoise2.fit()
print(egarchresult2.summary())

