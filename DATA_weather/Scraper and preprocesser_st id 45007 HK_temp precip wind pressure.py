import pandas as pd
import numpy as np
import meteostat 
from meteostat import Stations, Daily
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import cvxpy as cp
import warnings

########### available stations ########
import pandas as pd
from meteostat import Stations, Daily
from datetime import datetime

def list_available_stations(start_date, end_date):
    """
    Lists all available weather stations in Hong Kong with data coverage for the specified date range.
    
    Parameters:
    - start_date (datetime): Start date for data retrieval.
    - end_date (datetime): End date for data retrieval.
    
    Returns:
    - pd.DataFrame: DataFrame containing station details and data availability.
    """
    # Define the Hong Kong region (ISO code: HK)
    hk_region = 'HK'
    
    # Initialize Stations object
    stations = Stations()
    stations = stations.region(hk_region)
    
    # Fetch all stations in Hong Kong
    hk_stations = stations.fetch()
    
    if hk_stations.empty:
        print("No stations found in the Hong Kong region.")
        return pd.DataFrame()
    
    # Prepare a list to store station data coverage
    coverage_list = []
    
    for station_id, station in hk_stations.iterrows():
        # Fetch daily data for the station
        data = Daily(station_id, start_date, end_date)
        data = data.fetch()
        
        # Check data availability
        if not data.empty:
            available_days = len(data)
            total_days = (end_date - start_date).days + 1
            coverage = (available_days / total_days) * 100
            coverage_list.append({
                'station_id': station_id,
                'name': station['name'],
                'country': station['country'],
                'latitude': station['latitude'],
                'longitude': station['longitude'],
                'coverage_percent': coverage
            })
    
    # Create DataFrame
    coverage_df = pd.DataFrame(coverage_list)
    
    # Sort by coverage percentage in descending order
    coverage_df.sort_values(by='coverage_percent', ascending=False, inplace=True)
    
    return coverage_df

def main():
    # Define the time period for data retrieval
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    print("Listing available weather stations in Hong Kong with data coverage...")
    coverage_df = list_available_stations(start_date, end_date)
    
    if coverage_df.empty:
        print("No available stations with data for the specified date range.")
    else:
        print("\nAvailable Stations:")
        print(coverage_df.to_string(index=False))
        
        # Optionally, save to CSV for reference
        coverage_df.to_csv('hk_stations_coverage.csv', index=False)
        print("\nStation coverage details saved to 'hk_stations_coverage.csv'.")

if __name__ == "__main__":
    main()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def get_hong_kong_weather_data(start_date, end_date):
    """
    Fetches daily weather data for Hong Kong using Meteostat.
    
    Parameters:
    - start_date (datetime): Start date for data retrieval.
    - end_date (datetime): End date for data retrieval.
    
    Returns:
    - pd.DataFrame: DataFrame containing daily weather data.
    """
    # Specify the Hong Kong Observatory station (Station ID: 45007)
    station_id = '45007'  # Hong Kong Observatory

    # Fetch daily data
    data = Daily(station_id, start_date, end_date)
    data = data.fetch()
    
    if data.empty:
        raise ValueError("No weather data found for the specified station and date range.")
    
    # Reset index to have 'time' as a column
    data.reset_index(inplace=True)
    
    return data

def preprocess_weather_data(data):
    """
    Preprocesses the weather data by handling missing values and creating relevant features.
    
    Parameters:
    - data (pd.DataFrame): Raw weather data.
    
    Returns:
    - pd.DataFrame: Cleaned and feature-engineered weather data.
    """
    # Select relevant columns, including wind variables
    # Meteostat provides: tavg, tmin, tmax, prcp, snow, wdir, wspd, wpgt, pres, tsun
    # We'll include wind direction (wdir), wind speed (wspd), and pressure (pres)
    weather_df = data[['time', 'tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']]
    
    # Handle missing values
    # Forward fill
    weather_df.fillna(method='ffill', inplace=True)
    # Backward fill for any remaining NaNs
    weather_df.fillna(method='bfill', inplace=True)
    
    # Check for any remaining NaNs
    if weather_df.isnull().values.any():
        # If still NaNs, fill with overall mean
        weather_df.fillna(weather_df.mean(), inplace=True)
    
    # Verify no NaNs remain
    assert not weather_df.isnull().values.any(), "There are still missing values in the data."
    
    # Feature Engineering:
    # Create a temperature anomaly feature
    weather_df['temp_anomaly'] = weather_df['tavg'] - weather_df['tavg'].mean()
    
    # Create wind speed anomaly feature
    weather_df['wspd_anomaly'] = weather_df['wspd'] - weather_df['wspd'].mean()
    
    # Optionally, create wind direction as a categorical variable or convert to sine and cosine components
    # Here, we'll convert wind direction to radians and then to sine and cosine for better modeling
    weather_df['wdir_rad'] = np.deg2rad(weather_df['wdir'])
    weather_df['wdir_sin'] = np.sin(weather_df['wdir_rad'])
    weather_df['wdir_cos'] = np.cos(weather_df['wdir_rad'])
    
    # Drop the original wind direction in degrees as we have sine and cosine components
    weather_df.drop(columns=['wdir', 'wdir_rad'], inplace=True)
    
    # Save the preprocessed data to CSV
    weather_df.to_csv('hk_weather_data.csv', index=False)
    
    return weather_df

def fit_arima_model(weather_df, order=(5,1,0)):
    """
    Fits an ARIMA model to the temperature anomaly data.
    
    Parameters:
    - weather_df (pd.DataFrame): Preprocessed weather data.
    - order (tuple): The (p,d,q) order of the ARIMA model.
    
    Returns:
    - ARIMAResults: Fitted ARIMA model.
    - pd.Series: Forecasted temperature anomalies.
    - pd.DataFrame: Confidence intervals for forecasts.
    """
    # Extract the temperature anomaly series
    temp_anomaly = weather_df['temp_anomaly']
    
    # Fit ARIMA model
    model = ARIMA(temp_anomaly, order=order)
    model_fit = model.fit()
    
    print("ARIMA Model Summary:")
    print(model_fit.summary())
    
    # Forecast future temperature anomalies
    forecast_steps = 30  # Forecasting next 30 days
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()
    
    return model_fit, forecast_mean, forecast_conf_int

def price_weather_derivatives(weather_df, forecast_mean, forecast_conf_int, risk_aversion=1.0):
    """
    Prices weather derivatives based on forecasted temperature anomalies.
    
    Parameters:
    - weather_df (pd.DataFrame): Preprocessed weather data.
    - forecast_mean (pd.Series): Forecasted temperature anomalies.
    - forecast_conf_int (pd.DataFrame): Confidence intervals for forecasts.
    - risk_aversion (float): Risk aversion parameter for optimization.
    
    Returns:
    - dict: Dictionary containing derivative pricing results.
    """
    # Assume derivative payoff is linked to temperature anomaly
    # Payoff = max(temp_anomaly - threshold, 0) for heating derivatives
    threshold = 0.0  # Zero anomaly
    
    # Combine historical and forecasted anomalies
    combined_anomaly = pd.concat([weather_df['temp_anomaly'], forecast_mean], ignore_index=True)
    
    # Calculate derivative payoffs
    combined_payoff = combined_anomaly.apply(lambda x: max(x - threshold, 0))
    
    # Split into historical and forecasted payoffs
    historical_payoff = combined_payoff.iloc[:len(weather_df)]
    forecast_payoff = combined_payoff.iloc[len(weather_df):]
    
    # Compute statistics
    expected_payoff = historical_payoff.mean()
    variance_payoff = historical_payoff.var()
    
    # Optimization: Maximize expected payoff - risk aversion * variance
    # Define variables
    w = cp.Variable()  # Weight of derivative
    
    # Define objective: Maximize expected payoff * w - risk_aversion * variance_payoff * w^2
    objective = cp.Maximize(w * expected_payoff - risk_aversion * variance_payoff * cp.square(w))
    
    # Define constraints (e.g., weight bounds)
    constraints = [
        w >= 0,     # No short selling
        w <= 10     # Example: Maximum 10x leverage
    ]
    
    # Solve optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    optimal_weight = w.value
    optimal_expected_payoff = optimal_weight * expected_payoff
    optimal_variance = (optimal_weight ** 2) * variance_payoff
    
    print("\nOptimal Derivative Position:")
    print(f"Weight: {optimal_weight:.4f}")
    print(f"Expected Payoff: {optimal_expected_payoff:.4f}")
    print(f"Variance of Payoff: {optimal_variance:.4f}")
    
    results = {
        'weight': optimal_weight,
        'expected_payoff': optimal_expected_payoff,
        'variance_payoff': optimal_variance,
        'forecast_mean': forecast_mean,
        'forecast_conf_int': forecast_conf_int
    }
    
    return results

def visualize_results(weather_df, model_fit, forecast_mean, forecast_conf_int, derivative_results):
    """
    Visualizes the weather data, ARIMA model forecasts, and derivative payoffs.
    
    Parameters:
    - weather_df (pd.DataFrame): Preprocessed weather data.
    - model_fit (ARIMAResults): Fitted ARIMA model.
    - forecast_mean (pd.Series): Forecasted temperature anomalies.
    - forecast_conf_int (pd.DataFrame): Confidence intervals for forecasts.
    - derivative_results (dict): Results from derivative pricing.
    """
    plt.figure(figsize=(14, 6))
    
    # Plot Historical Temperature Anomalies
    plt.plot(weather_df['time'], weather_df['temp_anomaly'], label='Historical Temp Anomaly')
    
    # Plot Forecasted Temperature Anomalies
    forecast_dates = pd.date_range(start=weather_df['time'].iloc[-1] + pd.Timedelta(days=1),
                                   periods=len(forecast_mean), freq='D')
    plt.plot(forecast_dates, forecast_mean, label='Forecasted Temp Anomaly', color='orange')
    
    # Plot Confidence Intervals
    plt.fill_between(forecast_dates,
                     forecast_conf_int.iloc[:, 0],
                     forecast_conf_int.iloc[:, 1],
                     color='orange', alpha=0.2, label='Confidence Interval')
    
    plt.axhline(0, color='red', linestyle='--', label='Threshold')
    plt.title('Temperature Anomalies and ARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel('Temperature Anomaly (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Derivative Payoffs
    plt.figure(figsize=(14, 6))
    
    # Historical Payoffs
    plt.plot(weather_df['time'], weather_df['temp_anomaly'].apply(lambda x: max(x - 0.0, 0)), label='Historical Payoff')
    
    # Forecasted Payoffs
    forecast_payoff = derivative_results['forecast_mean'].apply(lambda x: max(x - 0.0, 0))
    plt.plot(forecast_dates, forecast_payoff, label='Forecasted Payoff', color='green')
    
    plt.title('Derivative Payoffs Based on Temperature Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Payoff')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Visualization of Optimization Results
    print("\nDerivative Pricing Results:")
    print(f"Optimal Weight: {derivative_results['weight']:.4f}")
    print(f"Expected Payoff: {derivative_results['expected_payoff']:.4f}")
    print(f"Variance of Payoff: {derivative_results['variance_payoff']:.4f}")

def main():
    # Define the time period for data retrieval
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    print("Fetching weather data for Hong Kong...")
    try:
        weather_data = get_hong_kong_weather_data(start_date, end_date)
        print(f"Retrieved {len(weather_data)} days of data.")
    except ValueError as e:
        print(f"Error fetching data: {e}")
        return
    
    print("\nPreprocessing data...")
    weather_df = preprocess_weather_data(weather_data)
    print("Data preprocessing complete.")
    
    print("\nFitting ARIMA model to temperature anomalies...")
    # Determine ARIMA order (p,d,q). For simplicity, using (5,1,0)
    # In practice, use AIC/BIC or grid search to determine the best order
    model_fit, forecast_mean, forecast_conf_int = fit_arima_model(weather_df, order=(5,1,0))
    print("ARIMA model fitting complete.")
    
    print("\nPricing weather derivatives based on forecasts...")
    derivative_results = price_weather_derivatives(weather_df, forecast_mean, forecast_conf_int, risk_aversion=1.0)
    print("Derivative pricing complete.")
    
    print("\nVisualizing results...")
    visualize_results(weather_df, model_fit, forecast_mean, forecast_conf_int, derivative_results)
    print("Visualization complete.")

if __name__ == "__main__":
    main()
