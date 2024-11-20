# Temperature
# Define the grid layout
from scipy.stats import kurtosis, skew, norm
# Import ECDF
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
import pickle
import os
import os


# ---------------------- Data Loading and Preprocessing ---------------------------------------------------------------- #

parquet_file_path = '/content/drive/MyDrive/wd/massive data/merged_parquet_files/temp.parquet'

try:
    df = pd.read_parquet(parquet_file_path)
    print("Data loaded successfully. Here's a preview:")
    print(df.head())
except FileNotFoundError:
    print(f"File not found at {parquet_file_path}. Please check the path and try again.")
    sys.exit()

df['obstime'] = pd.to_datetime(df['obstime'])

geod = pd.read_csv('/content/drive/MyDrive/wd/massive data/lat lon stations.csv')
print(geod)

df['station_id'] = df['station_id'].astype(str)
geod['station_id'] = geod['station_id'].astype(str)

# merge
data = pd.merge(df, geod, on='station_id', how='left')

# Group by 'station_id' and 'obstime' and aggregate
data = df.groupby(['station_id', 'obstime']).agg({'Air Temperature in degree C': 'mean'}).reset_index()
data.rename(columns={'Air Temperature in degree C': 'temperature'}, inplace=True)

# Resample for each station separately
resampled_data = []
for station_id, station_data in data.groupby('station_id'):
    station_data.set_index('obstime', inplace=True)
    station_data = station_data.resample('2H').mean()
    station_data['temperature'] = station_data['temperature'].interpolate(method='linear', limit_direction='both')
    station_data['station_id'] = station_id  # Add station_id back
    resampled_data.append(station_data.reset_index())

# Combine all stations' data
data = pd.concat(resampled_data)

print("\nResampled Data Preview:")
print(data.head())


# ---------------------- Feature Engineering ------------------------------------------------------------------- #

data = data.reset_index()

# Create day index d(i) as day of the year with fractional part
# Fractional part represents the time within the day
data['day_of_year'] = data['obstime'].dt.dayofyear
data['hour'] = data['obstime'].dt.hour + data['obstime'].dt.minute / 60.0
data['d_i'] = data['day_of_year'] + data['hour'] / 24.0
data['season'] = data['obstime'].apply(lambda x: 0 if x.month in [12,1,2] else (1 if x.month in [3,4,5] else (2 if x.month in [6,7,8] else 3)))
# 0/1/2/3 - winter/spring/summer/autumn


# time and temperature
time = data['d_i'].values  
temp_data = data['temperature'].values

# Add a small constant to avoid log(0) 
epsilon = 1e-3
data['temperature_log'] = np.log(data['temperature'] + epsilon)
temp_log_data = data['temperature_log'].values


# Number of observations
N = len(temp_log_data)

# ---------------------- KS test ---------------------- #

from scipy.stats import kurtosis, skew, norm, kstest, jarque_bera
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os
from math import ceil

# Define output directory
output_dir = '/content/drive/MyDrive/wFTproject/Gpt/stats'
os.makedirs(output_dir, exist_ok=True)

# Unique station IDs and group them into chunks of 8
station_ids = data['station_id'].unique()
stations_per_plot = 8
n_groups = ceil(len(station_ids) / stations_per_plot)

stats_table = []

for group_idx in range(n_groups):
    # Select stations for this plot
    start_idx = group_idx * stations_per_plot
    end_idx = min(start_idx + stations_per_plot, len(station_ids))
    group_station_ids = station_ids[start_idx:end_idx]

    # Prepare grid layout for plots
    gs = gridspec.GridSpec(len(group_station_ids) * 2, 4, height_ratios=[4, 4] * len(group_station_ids), hspace=0.25)
    fig = plt.figure(figsize=(25, len(group_station_ids) * 6))

    for i, station in enumerate(group_station_ids):
        # Filter data for the station
        station_data = data[data['station_id'] == station]
        temperature_data = station_data['temperature'].dropna()

        # Compute statistics
        mu, std = temperature_data.mean(), temperature_data.std()
        kur = kurtosis(temperature_data, fisher=False)
        skw = skew(temperature_data)

        # KS Test
        ks_stat, p_val = kstest(temperature_data, 'norm', args=(mu, std))
        # Jarque-Bera Test
        jb_stat, jb_pval = jarque_bera(temperature_data)

        stats_table.append({
            'station': station,
            'mean': mu,
            'std': std,
            'kurtosis': kur,
            'skewness': skw,
            'ks_stat': ks_stat,
            'ks_pval': p_val,
            'jb_stat': jb_stat,
            'jb_pval': jb_pval
        })

        # Plot histogram
        ax1 = plt.subplot(gs[2 * i, :2])
        n, bins, patches = ax1.hist(temperature_data, bins=30, density=True, alpha=0.6, color='g')
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        ax1.plot(x, norm.pdf(x, mu, std), 'b-', lw=2, label="Normal Distribution")
        ax1.set_ylabel("Density")
        ax1.set_title(f"Station {station}: Histogram", fontsize=14)

        # Annotate with summary statistics
        stats_text = f"Mean: {mu:.2f}\nStd Dev: {std:.2f}\nKurtosis: {kur:.2f}\nSkewness: {skw:.2f}"
        ax1.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=11)

        # Plot ECDF and theoretical CDF
        ax2 = plt.subplot(gs[2 * i, 2:])
        ecdf = ECDF(temperature_data)
        ax2.plot(ecdf.x, ecdf.y, label="ECDF", color='g')
        ax2.plot(ecdf.x, norm.cdf(ecdf.x, loc=mu, scale=std), label="Normal CDF", color='b')
        ax2.set_title(f"Station {station}: Kolmogorov-Smirnov", fontsize=14)
        ks_text = f"KS Stat: {ks_stat:.3f}\nP-value: {p_val:.3f}"
        jb_text = f"JB Stat: {jb_stat:.3f}\nP-value: {jb_pval:.3f}"
        ax2.annotate(ks_text, xy=(0.95, 0.05), xycoords='axes fraction', ha='right', va='bottom', fontsize=11)
        ax2.annotate(jb_text, xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=11)
        ax2.legend()

        # Plot ACF
        ax3 = plt.subplot(gs[2 * i + 1, :])
        max_lags = min(40, len(temperature_data) - 1)  # Ensure lags do not exceed available data
        if max_lags > 0:
            plot_acf(temperature_data, ax=ax3, lags=max_lags, zero=False, alpha=0.05, title=f"Station {station}: Autocorrelation")
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for ACF', ha='center', va='center', fontsize=12)
            ax3.set_title(f"Station {station}: Autocorrelation")

    # Save the current plot to file
    plot_filename = os.path.join(output_dir, f"stations_group_{group_idx + 1}.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {plot_filename}")

# Save statistics table as a DataFrame
stats_df = pd.DataFrame(stats_table)

# Save as CSV and LaTeX
csv_path = os.path.join(output_dir, "stats_table.csv")
latex_path = os.path.join(output_dir, "stats_table.tex")

stats_df.to_csv(csv_path, index=False, float_format="%.2f")
latex_table = stats_df.to_latex(index=False, float_format="%.2f")
with open(latex_path, "w") as f:
    f.write(latex_table)

print(f"Saved statistics table to {csv_path} and {latex_path}.")


# ------------------------------- Time Series Functions --------------------------------#
import os
import sys
import pickle
from datetime import datetime, date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis, skew, norm
from scipy.optimize import minimize, curve_fit
from scipy import stats
from scipy.linalg import expm
from statsmodels.distributions.empirical_distribution import ECDF
from symfit import parameters, variables, sin, cos, Fit
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from math import ceil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def seasonality_simple(x, a0, a1):
    return a0 + a1 * x

def seasonality_benth(x, a_0, a_1, a_2, a_3):
    omega = 2 * np.pi / 365.25
    y_pred = a_0 + a_1 * x + a_2 * np.cos(omega * x) + a_3 * np.sin(omega * x)
    return y_pred

def seasonality_cabrera(x, a_0, a_1, a_2, a_3, d_1, d_2):
    omega = 2 * np.pi / 365.25
    y_pred = (
        a_0 +
        a_1 * x +
        a_2 * np.cos(omega * (x - d_1)) +
        a_3 * np.sin((omega / 2) * (x - d_2))
    )
    return y_pred

def RSS(y, y_pred):
    return np.sqrt(np.sum((y - y_pred) ** 2))

def fourier_series(x, f, n=0):
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    series = a0 + sum(
        ai * cos(i * f * x) + bi * sin(i * f * x)
        for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1)
    )
    return series

def autoregressive_time_varying_variance_fit(df, seasonality_mean_func, variable='temperature', max_ar_lag=3, fourier_order=4):
    # Convert the index to a numerical representation, e.g., ordinal values
    x_data = df.index.map(datetime.toordinal).to_numpy()  # Convert DatetimeIndex to ordinal
    y_data = df[variable].to_numpy()  # Ensure y_data is a numpy array

    # Fit the seasonality function
    try:
        params, cov = curve_fit(seasonality_mean_func, x_data, y_data, method='lm')
    except Exception as e:
        logging.error(f"Curve fitting failed: {e}")
        raise

    # Seasonalize the variable
    seasonal_pred = seasonality_mean_func(x_data, *params)
    df['seasonalized'] = y_data - seasonal_pred

    # Select the optimal lag
    try:
        optimal = ar_select_order(df['seasonalized'], maxlag=max_ar_lag, ic='aic', trend='n')
    except Exception as e:
        logging.error(f"AR order selection failed: {e}")
        raise

    # Fit the model using AutoReg
    try:
        model = AutoReg(df['seasonalized'], lags=optimal.ar_lags, trend='n')
        model_fit = model.fit()
    except Exception as e:
        logging.error(f"AutoReg fitting failed: {e}")
        raise

    # Add the constant variance residuals to the dataframe
    df['residuals_cv'] = model_fit.resid

    # Define Fourier series using symfit
    x, y = variables('x, y')
    fourier_series_model = fourier_series(x, f=2 * np.pi / 365.25, n=fourier_order)
    fourier_model_dict = {y: fourier_series_model}
    time_shift = np.max(optimal.ar_lags)

    # Convert x and y to numpy arrays for Fourier fitting
    x_values = df[time_shift:].index.map(datetime.toordinal).to_numpy()  # Convert index to ordinal values
    y_values = (df[time_shift:]['residuals_cv']**2).to_numpy()  # Convert residuals to numpy array

    try:
        fourier_model = Fit(fourier_model_dict, x=x_values, y=y_values)
        fourier_model_fit = fourier_model.execute()
    except Exception as e:
        logging.error(f"Fourier model fitting failed: {e}")
        raise

    # Add the time-varying variance residuals to the dataframe
    fitted_variance = fourier_model_fit.model(x=df.index.map(datetime.toordinal).to_numpy(), **fourier_model_fit.params).y
    df['fitted_variance'] = fitted_variance
    df['residuals_tv'] = df['residuals_cv'] / np.sqrt(df['fitted_variance'])

    return {
        'seasonality_function': (params, cov),
        'ar_model': model_fit,
        'fourier_model': fourier_model_fit,
        'result': df
    }

def autoregressive_time_varying_variance_fit(df, seasonality_mean_func, variable='temperature', max_ar_lag=3, fourier_order=4):
    # Convert the index to a numerical representation, e.g., ordinal values
    x_data = df.index.map(datetime.toordinal).to_numpy()  # Convert DatetimeIndex to ordinal
    y_data = df[variable].to_numpy()  # Ensure y_data is a numpy array

    # Fit the seasonality function
    try:
        params, cov = curve_fit(seasonality_mean_func, x_data, y_data, maxfev=5000, method='lm')
    except Exception as e:
        logging.error(f"Curve fitting failed: {e}")
        raise
    
    # Seasonalize the variable
    seasonal_pred = seasonality_mean_func(x_data, *params)
    df['seasonalized'] = y_data - seasonal_pred

    # Select the optimal lag
    try:
        optimal = ar_select_order(df['seasonalized'], maxlag=max_ar_lag, ic='aic', trend='n')
    except Exception as e:
        logging.error(f"AR order selection failed: {e}")
        raise
    
    # Fit the model using AutoReg
    try:
        model = AutoReg(df['seasonalized'], lags=optimal.ar_lags, trend='n')
        model_fit = model.fit()
    except Exception as e:
        logging.error(f"AutoReg fitting failed: {e}")
        raise

    # Add the constant variance residuals to the dataframe
    df['residuals_cv'] = model_fit.resid

    # Define Fourier series using symfit
    x, y = variables('x, y')
    fourier_series_model = fourier_series(x, f=2 * np.pi / 365.25, n=fourier_order)
    fourier_model_dict = {y: fourier_series_model}
    time_shift = np.max(optimal.ar_lags)
    
    # Convert x and y to numpy arrays for Fourier fitting
    x_values = df[time_shift:].index.map(datetime.toordinal).to_numpy()  # Convert index to ordinal values
    y_values = (df[time_shift:]['residuals_cv']**2).to_numpy()  # Convert residuals to numpy array
    
    try:
        fourier_model = Fit(fourier_model_dict, x=x_values, y=y_values)
        fourier_model_fit = fourier_model.execute()
    except Exception as e:
        logging.error(f"Fourier model fitting failed: {e}")
        raise
    
    # Add the time-varying variance residuals to the dataframe
    df['fitted_variance'] = fourier_model_fit.model(x=df.index.map(datetime.toordinal).to_numpy(), **fourier_model_fit.params).y
    df['residuals_tv'] = df['residuals_cv'] / np.sqrt(df['fitted_variance'])
    
    return {
        'seasonality_function': (params, cov),
        'ar_model': model_fit,
        'fourier_model': fourier_model_fit,
        'result': df
    }



#---------------------------- CAR Functions -------------------------------------#

def ar_car_link(phi):
    phi = np.asarray(phi)  # Ensure phi is a NumPy array
    p = len(phi)
    if p == 1:
        x = [1 - phi[0]]
    elif p == 2:
        A_matrix = np.array([[-1, 0], [1, -1]])
        b_vector = np.array([phi[0] - 2, phi[1] + 1])
        x = np.linalg.solve(A_matrix, b_vector)
    elif p == 3:
        A_matrix = np.array([[-1, 0, 0], [2, -1, 0], [-1, 1, -1]])
        b_vector = np.array([phi[0] - 3, phi[1] + 3, phi[2] - 1])
        x = np.linalg.solve(A_matrix, b_vector)
    elif p == 4:
        A_matrix = np.array([[1, 0, 0, 0], [3, -1, 0, 0], [-3, 2, -1, 0], [1, -1, 1, -1]])
        b_vector = np.array([-phi[0] + 4, phi[1] + 6, phi[2] - 4, phi[3] + 1])
        x = np.linalg.solve(A_matrix, b_vector)
    else:
        raise ValueError('Only p=1,2,3,4 supported')
    
    # CAR(p) model
    A = np.zeros((p, p))
    for i in range(p):
        if i > 0:
            A[i-1, i] = 1
        A[p-1, i] = -x[len(x)-i-1]
    
    return A, x


class EulerSchemeCAR:
    def __init__(self, A, seasonality_mean_func, seasonality_mean_params, variance_func, variance_func_params, dt=1):
        self.A = A
        self._ = expm(A * dt)
        
        self.variance_func = variance_func
        self.variance_func_params = variance_func_params
        self.seasonality_mean_func = seasonality_mean_func
        self.seasonality_mean_params = seasonality_mean_params
        
        self.variance = None
        self.mean = None
        
        self.dt = dt
        self.p = A.shape[0]
        # Euclidean unit vector
        e = np.zeros(self.p)
        e[-1] = 1
        self.e = e.reshape(self.p, 1)
    
    def simulate(self, x0, t0, t1, nsim=100):
        n_steps = int((t1 - t0) / self.dt)
        time_points = np.linspace(t0, t1, n_steps)
        variance = self.variance_func.model(x=time_points, **self.variance_func_params).y
        mean = self.seasonality_mean_func(time_points, *self.seasonality_mean_params)
        
        xs_sim = np.zeros((nsim, self.p, n_steps))
        
        for s in range(nsim):
            xs = np.zeros((n_steps, self.p))
            xs[0] = np.full(self.p, x0)
            for i in range(1, n_steps):
                xs[i] = self.euler_step(xs[i-1], variance[i])
            xs_sim[s] = xs.T
        
        b = np.zeros(self.p).reshape(1, self.p)
        b[0] = 1
        ys_sim = np.dot(b, xs_sim)
        ms_sim = ys_sim + mean
        return xs_sim.transpose(), ys_sim.transpose()[:, :, 0], ms_sim.transpose()[:, :, 0]
    
    def euler_step(self, x, v):
        epsilon = np.random.normal(0, 1, size=1)
        x = np.matmul(self._, x) + np.matmul(self._, self.e).dot(epsilon * np.sqrt(v) * np.sqrt(self.dt))
        return x


#------------------------------------------ Using simulations to compute the weather derivatives ---------------------------------#

# HDD(\tau_{1},\tau_{2}) = \int_{\tau_{1}}^{\tau_{2}} \max(c - T_{u}, 0) du,
def HDD(x, t0, t1 , c = 18.0, y = []):
    # x is a vector of (simulated) temperatures, if its a matrix return a vector of columns
    # y is a vector of actual temperatures
    # c is the baseline temperature
    _x = x[t0:t1].copy()
    _y = y[t0:t1].copy()

    # Replace the simulated temperatures with the actual temperatures (if available)
    if len(_y) > 0:
        # Fill a matrix of the same size as x with the values of y
        _x[:len(_y)] = np.tile(_y, (_x.shape[1], 1)).T
        
    return np.sum(np.maximum(c - _x, 0), axis = 0)
    
# CDD(\tau_{1},\tau_{2}) = \int_{\tau_{1}}^{\tau_{2}} \max(T_{u}-c, 0) du,
def CDD(x, t0, t1 , c = 18.0, y = []):
    # x is a vector of (simulated) temperatures, if its a matrix return a vector of columns
    # y is a vector of actual temperatures
    # c is the baseline temperature
    _x = x[t0:t1].copy()
    _y = y[t0:t1].copy()

    # Replace the simulated temperatures with the actual temperatures (if available)
    if len(_y) > 0:
        # Fill a matrix of the same size as x with the values of y
        _x[:len(_y)] = np.tile(_y, (_x.shape[1], 1)).T
        
    return np.sum(np.maximum(_x - c, 0), axis = 0)

# CAT(\tau_{1},\tau_{2}) = \int_{\tau_{1}}^{\tau_{2}} T_{u} du,
def CAT(x, t0, t1, y = []):
    # x is a vector of (simulated) temperatures, if its a matrix return a vector of columns
    # y is a vector of actual temperatures
    # c is the baseline temperature
    _x = x[t0:t1].copy()
    _y = y[t0:t1].copy()

    # Replace the simulated temperatures with the actual temperatures (if available)
    if len(_y) > 0:
        # Fill a matrix of the same size as x with the values of y
        _x[:len(_y)] = np.tile(_y, (_x.shape[1], 1)).T
        
    return np.sum(_x, axis = 0)


#---------------------------------Indices calculator -------------------------------#

'''

monthly HDD, CAT running from:
OCT, NOV, DEC, JAN, FEB, MAR: HDD
APR, MAY, JUN, JUL, AUG, SEP, OCT: CAT

'''
import pandas as pd

def actual_temperature_indices(car, ar_fit, nsim=100, index_points=20):
    df_actual = ar_fit['result']  # The dataframe with the actual temperatures
    start_date = df_actual.index.min()
    end_date = df_actual.index.max()

    # Get the positions of the start and end date in df_actual
    start_idx = df_actual.index.get_loc(start_date)
    end_idx = df_actual.index.get_loc(end_date)

    # Simulate the CAR(p) model
    xs, ys, ms = car.simulate(x0=1, t0=start_idx, t1=end_idx, nsim=nsim)

    # Strip - Monthly
    df_actual['year'] = df_actual.index.year
    df_actual['month'] = df_actual.index.month
    df_actual['season'] = df_actual.index.month.map(
        lambda x: 0 if x in [12, 1, 2] else (1 if x in [3, 4, 5] else (2 if x in [6, 7, 8] else 3))
    )

    # Initialize an empty list to store the results
    results = []

    # For each granularity, calculate the CAR(p) values
    for (year, month), group in df_actual.groupby(['year', 'month']):
        indices = group.index
        for idx in indices:
            idx_position = df_actual.index.get_loc(idx)
            if idx_position < start_idx:
                continue

            # Calculate slice indices for ms
            slice_start = df_actual.index.get_loc(indices[0]) - start_idx
            slice_end = idx_position - start_idx

            # Ensure slice indices are within bounds
            if slice_start < 0:
                slice_start = 0
            if slice_end < slice_start:
                slice_end = slice_start

            ms_selected = ms[slice_start:slice_end]

            # Slice actual_selected using the previous index
            if idx_position == 0:
                actual_selected = pd.Series([], dtype='float')
            else:
                previous_idx = df_actual.index[idx_position - 1]
                actual_selected = df_actual.loc[indices[0]:previous_idx, 'temperature']

            # Calculate the CAT, CDD, and HDD values for each day of the month
            if ms_selected.size == 0:
                cat_value = np.nan
                cdd_value = np.nan
                hdd_value = np.nan
            else:
                cat_value = CAT(x=ms_selected, y=actual_selected, t0=0, t1=len(ms_selected)).mean()
                cdd_value = CDD(x=ms_selected, y=actual_selected, t0=0, t1=len(ms_selected)).mean()
                hdd_value = HDD(x=ms_selected, y=actual_selected, t0=0, t1=len(ms_selected)).mean()

            # Determine the Overall Temperature (OT) value
            ot_value = cat_value if month in [5, 6, 7, 8, 9] else hdd_value

            # Prepare the result for the current index
            result_entry = {
                'date': idx,  # Use idx directly
                'year': year,
                'month': month,
                f'strip_cat_{month}': cat_value * index_points if not np.isnan(cat_value) else np.nan,
                f'strip_cdd_{month}': cdd_value * index_points if not np.isnan(cdd_value) else np.nan,
                f'strip_hdd_{month}': hdd_value * index_points if not np.isnan(hdd_value) else np.nan,
                'strip_cat_continuous': cat_value * index_points if not np.isnan(cat_value) else np.nan,
                'strip_cdd_continuous': cdd_value * index_points if not np.isnan(cdd_value) else np.nan,
                'strip_hdd_continuous': hdd_value * index_points if not np.isnan(hdd_value) else np.nan,
                'strip_ot_continuous': ot_value * index_points if not np.isnan(ot_value) else np.nan,
                'temperature': group.loc[idx, 'temperature'],
                'ms': ms_selected.mean() if ms_selected.size > 0 else np.nan
            }

            # Store the result
            results.append(result_entry)

    # Convert the results to a DataFrame and handle NaNs
    df_results = pd.DataFrame(results).bfill()
    df_results['date'] = pd.to_datetime(df_results['date'], format='%Y-%m-%d')

    # Calculate the payoff for each instrument
    for instrument in ['cat', 'cdd', 'hdd', 'ot']:
        df_results[f'strip_{instrument}_continuous_payoff'] = df_results[f'strip_{instrument}_continuous'].diff()
        # Each first day of the month will have a payoff of 0
        df_results.loc[df_results['date'].dt.day == 1, f'strip_{instrument}_continuous_payoff'] = 0

    return df_results


#---------------------------- Main Processing -------------------------------------#

import os
import sys
import pickle
from datetime import datetime, date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis, skew, norm
from scipy.optimize import minimize, curve_fit
from scipy import stats
from scipy.linalg import expm
from statsmodels.distributions.empirical_distribution import ECDF
from symfit import parameters, variables, sin, cos, Fit
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from math import ceil
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

MIN_DATA_POINTS = 6

# ---------------------- Data Loading and Preprocessing ---------------------- #

parquet_file_path = '/content/drive/MyDrive/wd/massive data/merged_parquet_files/temp.parquet'

# Load the parquet file
try:
    data = pd.read_parquet(parquet_file_path)
    logging.info("Data loaded successfully. Here's a preview:")
    logging.debug(f"\n{data.head()}")
except FileNotFoundError:
    logging.error(f"File not found at {parquet_file_path}. Please check the path and try again.")
    sys.exit()

# Ensure 'obstime' is in datetime format
data['obstime'] = pd.to_datetime(data['obstime'])

# Group by 'station_id' and 'obstime' and aggregate
data = data.groupby(['station_id', 'obstime']).agg({'Air Temperature in degree C': 'mean'}).reset_index()
data.rename(columns={'Air Temperature in degree C': 'temperature'}, inplace=True)

# Resample for each station separately
resampled_data = []
for station_id, station_data in data.groupby('station_id'):
    station_data = station_data.set_index('obstime').resample('h').agg({'temperature': 'mean'})
    # Handle duplicates by taking the first occurrence if any
    if station_data.index.duplicated().any():
        station_data = station_data[~station_data.index.duplicated(keep='first')]
    # Interpolate missing values
    station_data['temperature'] = station_data['temperature'].interpolate(method='linear', limit_direction='both')
    station_data['station_id'] = station_id  # Add station_id back
    resampled_data.append(station_data.reset_index())

# Combine all stations' data
data = pd.concat(resampled_data)
logging.info("\nResampled Data Preview:")
logging.debug(f"\n{data.head()}")

# ---------------------- Feature Engineering ---------------------- #

data = data.reset_index(drop=True)

# Create day index d(i) as day of the year with fractional part
data['day_of_year'] = data['obstime'].dt.dayofyear
data['hour'] = data['obstime'].dt.hour + data['obstime'].dt.minute / 60.0
data['d_i'] = data['day_of_year'] + data['hour'] / 24.0
data['season'] = data['obstime'].apply(
    lambda x: 0 if x.month in [12, 1, 2] else (1 if x.month in [3, 4, 5] else (2 if x.month in [6, 7, 8] else 3))
)

# Time and temperature
time = data['d_i'].values
temp_data = data['temperature'].values

# Add a small constant to avoid log(0)
epsilon = 1e-3
data['temperature_log'] = np.log(data['temperature'] + epsilon)
temp_log_data = data['temperature_log'].values

# ------------------------------- Main Loop for Processing Stations -------------------------------- #

CAR_temperature_dict = {}

# Define output path
output_dir = '/content/drive/MyDrive/wFTproject/Gpt/model_outputs'
output_file = os.path.join(output_dir, 'CAR_temperature_dict_f.pkl')

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over each station and process
for station_id in data['station_id'].unique():
    logging.info(f"Processing Station ID: {station_id}")
    
    # Extract temperature data for the station
    temperature_df = data[data['station_id'] == station_id][['obstime', 'temperature']].copy()
    
    if temperature_df.empty:
        logging.warning(f"No data available for Station ID {station_id}. Skipping.")
        continue

    # Keep 'obstime' as a DatetimeIndex
    temperature_df = temperature_df.sort_values(by='obstime').set_index('obstime')

    # **Daily Resampling**
    temperature_df = temperature_df.resample('D').mean()
    temperature_df['temperature'] = temperature_df['temperature'].interpolate(method='linear', limit_direction='both')

    # **Check for Sufficient Data Points**
    if len(temperature_df) < MIN_DATA_POINTS:
        logging.warning(f"Station ID {station_id} has only {len(temperature_df)} data points. Skipping curve fitting.")
        continue

    # **Select Seasonality Function Based on Data Size**
    if len(temperature_df) >= 6:
        seasonality_func = seasonality_cabrera
    else:
        seasonality_func = seasonality_simple  

    # **Fit the Autoregressive Model**
    try:
        station_result = autoregressive_time_varying_variance_fit(
            temperature_df, 
            seasonality_cabrera, 
            variable='temperature', 
            max_ar_lag=3, 
            fourier_order=4
        )
        logging.info(f"Model fitted successfully for Station ID: {station_id}")
    except Exception as e:
        logging.error(f"Error fitting model for Station ID {station_id}: {e}")
        continue

    # Get the link between the AR model and the CAR model
    try:
        ar_model = station_result['ar_model']
        A, x = ar_car_link(ar_model.params)
        variance_func = station_result['fourier_model']
        variance_func_params = station_result['fourier_model'].params
        mean_func = seasonality_cabrera
        mean_func_params = station_result['seasonality_function'][0]

        # Create the CAR model
        station_CAR = EulerSchemeCAR(
            A, 
            seasonality_mean_func=mean_func, 
            seasonality_mean_params=mean_func_params, 
            variance_func=variance_func, 
            variance_func_params=variance_func_params
        )

        # Save the CAR model in the dictionary
        CAR_temperature_dict[station_id] = {
            'car_model': station_CAR,
            'ar_model': station_result,
            'station_id': station_id,
        }

        # **Simulation and Index Calculation**
        logging.info(f"Running simulation for Station ID: {station_id}")
        station_indices = actual_temperature_indices(station_CAR, station_result, nsim=250)
        CAR_temperature_dict[station_id]['temperature_indices'] = station_indices
        logging.info(f"Simulation and index calculation completed for Station ID: {station_id}")

    except Exception as e:
        logging.error(f"Error during CAR model creation or simulation for Station ID {station_id}: {e}")
        continue

    # Save progress after each station is processed
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(CAR_temperature_dict, f)
        logging.info(f"Updated CAR_temperature_dict saved to {output_file} after processing Station ID: {station_id}")
    except Exception as e:
        logging.error(f"Failed to save CAR_temperature_dict after processing Station ID {station_id}: {e}")

# -------------------------------- Summary -------------------------------- #
logging.info(f"Finished processing {len(CAR_temperature_dict)} stations successfully.")


# -------------------------------- Plot - to add -------------------------------- #

