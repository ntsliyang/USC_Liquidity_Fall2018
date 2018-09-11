import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA


series = pd.read_csv('Microsoft Share Volume.csv', index_col='date', parse_dates=True)
print(series)


# series_log=np.log(series)
# print(series_log)
#
# difference=series_log-series_log.shift()
# print(difference)
#
series.plot()
plt.show()

plot_acf(series, lags=10)
plt.show()

plot_pacf(series, lags=10)
plt.show()



# mod=ARMA(difference, order=(1, 1))
# result=mod.fit()
# print(result.summary())
#
# mod2=ARMA(series, order=(3, 2))
# result2=mod2.fit()
# print(result2.summary())
#
# series2=pd.read_csv("Microsoft Share Volume.csv", usecols=[1])
# print(series2)
#
# mod3=ARMA(series2, order=(3, 2))
# result3=mod3.fit()
# print(result3.summary())
# result3.plot_predict(start=2000, end=2800)
# plt.show()
#
# series3=pd.read_csv("Microsoft Share Volume Monthly.csv", index_col='date', parse_dates=True)
# month=series3.to_period(freq="M")
# print(month)
#
# month.plot()
# plt.show()
#
# # plot_acf(month, lags=10)
# # plt.show()
# #
# # plot_pacf(month, lags=10)
# # plt.show()
#
# mod4=ARMA(month, order=(1, 1))
# result4=mod4.fit()
# print(result4.summary())
# result4.plot_predict(start='2000-01', end='2018-12')
# plt.show()



# look at differences and rerun analysis
# use small-cap stock
# add EGARCH assumption
