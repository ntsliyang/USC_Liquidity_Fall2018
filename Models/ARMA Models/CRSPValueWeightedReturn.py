import pandas as pd
from pandas import Series
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima_model import ARMA
import numpy as np

series = Series.from_csv('CRSP value weighted return 2.csv', header=0)
print(series)
# series.plot()
# plt.show()
#
# plot_acf(series, lags=10)
# plt.show()
#
# plot_pacf(series, lags=10)
# plt.show()
#
BIC=np.zeros(7)
print(BIC)
print("Hello, please look at this/n/n/n/n")
for p in range(7):
    mod=ARMA(series, order=(p, 0))
    res=mod.fit()
    BIC[p]=res.bic
#
plt.plot(range(1, 7), BIC[1:7], marker="o")
plt.xlabel('Order of AR Model')
plt.ylabel('Bayesian Information Criterion')
plt.show()



#
# mod1=ARMA(series, order=(1, 0))
# result1=mod1.fit()
# print(result1.summary())
#
# mod2=ARMA(series, order=(2, 0))
# result2=mod2.fit()
# print(result2.summary())

# result1.plot_predict('20170101', '20180101')
# plt.show()

