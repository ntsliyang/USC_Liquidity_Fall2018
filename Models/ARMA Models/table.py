from numpy import *
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARMAResults
from pandas import Series


x=range(16)
x=reshape(x, (4, 4))

for p in range(4):
    for q in range(4):
        series = Series.from_csv('Microsoft Share Volume.csv', header=0)
        mod=ARMA(series, order=(p, q))
        res=mod.fit()
        x[p][q]=res.bic

print(x)


