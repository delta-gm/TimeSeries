import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, pacf, graphics

sns.set_style('darkgrid')
pd.plotting.register_matplotlib_converters()
# Default figure size
sns.mpl.rc('figure',figsize=(16, 6))

'''
The first set of examples uses the month-over-month growth rate in U.S. Housing starts that has not been seasonally adjusted. 
The seasonality is evident by the regular pattern of peaks and troughs. We set the frequency for the time series to “MS” (month-start) to avoid warnings when using AutoReg.
'''
data = pdr.get_data_fred('HOUSTNSA', '1959-01-01', '2019-06-01')
housing = data.HOUSTNSA.pct_change().dropna()
# Scale by 100 to get percentages
housing = 100 * housing.asfreq('MS')
fig, ax = plt.subplots()
ax = housing.plot(ax=ax)

'''
We can start with an AR(3). While this is not a good model for this data, it demonstrates the basic use of the API.
'''
mod = AutoReg(housing, 3, old_names=False)
res = mod.fit()
#print(res.summary())

'''
AutoReg supports the same covariance estimators as OLS. Below, we use cov_type="HC0", which is White’s covariance estimator. 
While the parameter estimates are the same, all of the quantities that depend on the standard error change.
'''
res = mod.fit(cov_type="HC0")
#print(res.summary())

sel = ar_select_order(housing, 13, old_names=False)
sel.ar_lags
res = sel.model.fit()
#print(res.summary())

#fig = res.plot_predict(720, 840)
# fig = plt.figure(figsize=(16,9))
# fig = res.plot_diagnostics(fig=fig, lags=30)

sel = ar_select_order(housing, 13, seasonal=True, old_names=False)
sel.ar_lags
res = sel.model.fit()
print(res.summary())

yoy_housing = data.HOUSTNSA.pct_change(12).resample("MS").last().dropna()
_, ax = plt.subplots()
ax = yoy_housing.plot(ax=ax)

sel = ar_select_order(yoy_housing, 13, old_names=False)
sel.ar_lags

sel = ar_select_order(yoy_housing, 13, glob=True, old_names=False)
sel.ar_lags
res = sel.model.fit()
print(res.summary())

fig = plt.figure(figsize=(16,9))
fig = res.plot_diagnostics(fig=fig, lags=30)



plt.show()