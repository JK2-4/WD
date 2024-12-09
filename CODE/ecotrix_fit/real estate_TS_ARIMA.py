# ARIMA(3,0,2) on log differenced fits

import matplotlib.pylab as plt
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from tabulate import tabulate
sns.set(style='whitegrid')


re1 = pd.read_csv('/content/drive/MyDrive/wFTproject/Weathermodel/re1.csv')
re1.head()
t = re1[re1['Code'] == 'CW0060']
t['DateValue'] = pd.to_datetime(t['DateValue'].astype(str), format='%Y%m')
t.set_index('DateValue', inplace=True)
# t.index check

#!pip install pmdarima --quiet
def adf_test(series, title=''):
    """
    Perform Augmented Dickey-Fuller test and print results.
    
    Parameters:
    - series (pd.Series): Time series data.
    - title (str): Title for the test.
    
    Returns:
    - dict: Test statistics and p-value.
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series, autolag='AIC')
    labels = ['Test Statistic','p-value','# Lags Used','Number of Observations Used']
    out = pd.Series(result[0:4], index=labels)
    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
    print(out.to_frame(name='Value'))
    
    if result[1] <= 0.05:
        print("=> The series is stationary.")
    else:
        print("=> The series is non-stationary.")
    
    return {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Used Lags': result[2],
        'Number of Observations': result[3],
        'Critical Values': result[4],
        'Stationary': result[1] <= 0.05
    }
def kpss_test(series, title=''):
    """
    Perform KPSS test and print results.
    
    Parameters:
    - series (pd.Series): Time series data.
    - title (str): Title for the test.
    
    Returns:
    - dict: Test statistics and p-value.
    """
    print(f'KPSS Test: {title}')
    statistic, p_value, lags, critical_values = kpss(series, regression='c')
    print(f'Test Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'Critical Values:')
    for key, value in critical_values.items():
        print(f'  {key} : {value}')
    
    if p_value < 0.05:
        print("=> The series is non-stationary.\n")
    else:
        print("=> The series is stationary.\n")
    
    return {
        'Test Statistic': statistic,
        'p-value': p_value,
        'Lags Used': lags,
        'Critical Values': critical_values,
        'Stationary': p_value >= 0.05
    }

# differencing: I(d) and log transformation : log(Yt - Yt-1 / Yt)
xx = ((np.log(t.edu_ter).diff())).replace([np.inf, -np.inf], np.nan).dropna() #log(Yt/Yt-1) shows heteroskedacticity and autocorrelation upto lag 10
xx = xx.reindex_like(t, method=None)
xx = xx.dropna()
xx.index
print('SHAPE OF SERIES IS ',xx.shape)

plt.figure(figsize=(15,7))
plt.title("Log differenecd rent")
plt.xlabel('Date')
plt.ylabel('xx')
plt.plot(xx)
plt.show()

seasonal_decompose(xx, model='additive', filt=None, period=12, two_sided=True, extrapolate_trend=0).plot();

print('lags are', acf(xx))
#ACF - MA?
plot_acf(xx)
plot_pacf(xx, method='ywm')

adf_result = adf_test(xx, title='Second-Order Differenced Log(Edu_Ter)')
kpss_result = kpss_test(xx, title='Second-Order Differenced Log(Edu_Ter)')

# Validate the results
if adf_result['Stationary'] and kpss_result['Stationary']:
    print("The series is confirmed to be stationary based on both ADF and KPSS tests.")
elif adf_result['Stationary'] and not kpss_result['Stationary']:
    print("ADF test indicates stationarity, but KPSS test does not.")
elif not adf_result['Stationary'] and kpss_result['Stationary']:
    print("KPSS test indicates stationarity, but ADF test does not.")
else:
    print("The series is non-stationary based on both ADF and KPSS tests.")

model = ARIMA(xx, order=(3, 0, 2), enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

residuals = model_fit.resid

plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of ARIMA(3, 0, 2) Model')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.grid(True)
plt.show()

plot_acf(residuals)
plot_pacf(residuals, method='ywm')

lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True) # LB TEST ON residuals - should be normal (null is no autocorrel (indep resid))
print("\nLjung-Box Test Results:")
print(lb_test)

forecast = model_fit.get_forecast(steps=12)
forecast_ci = forecast.conf_int()
forecast_mean = forecast.predicted_mean
plt.figure(figsize=(12, 6))
plt.plot(xx, label='Observed')
plt.plot(forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, 
                 forecast_ci['lower edu_ter'], 
                 forecast_ci['upper edu_ter'], 
                 color='pink', alpha=0.3)
plt.title('Forecast')
plt.xlabel('Date')
plt.ylabel('xx')
plt.legend()
plt.grid(True)
plt.show()

#SARIMAX?
