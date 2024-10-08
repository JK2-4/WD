###### Guassian Copula, Equal weighted index and Power Utility Bounds priced

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.optimize import root_scalar
from scipy.stats import norm, t, expon, gamma
from mpl_toolkits.mplot3d import Axes3D

# Assuming pyraingen is a valid package installed in your environment
import pyraingen


# Set random seed for reproducibility
np.random.seed(42)

# Directory to save the results
save_dir = 'results/'
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# 1. Simulate Weather Variables
# ----------------------------

def simulate_rainfall(n_sim, n_steps, params):
    """
    Simulate daily rainfall data using pyraingen.
    """
    # Placeholder: Replace with actual pyraingen simulation code
    # For demonstration, we'll simulate rainfall as random variables
    # In practice, use regionaliseddailysim or other appropriate functions
    rainfall = np.random.gamma(shape=params['rain_shape'], scale=params['rain_scale'], size=(n_sim, n_steps))
    return rainfall

def simulate_temperature(n_sim, n_steps, dt, params):
    """
    Simulate temperature using a mean-reverting Ornstein-Uhlenbeck (OU) model.
    """
    theta = params['theta']  # Speed of reversion
    mu = params['mu']        # Long-term mean
    sigma = params['sigma']  # Volatility

    temperature = np.zeros((n_sim, n_steps))
    temperature[:, 0] = params['T0']

    for t in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt), n_sim)
        temperature[:, t] = temperature[:, t-1] + theta * (mu - temperature[:, t-1]) * dt + sigma * dW

    return temperature

def simulate_wind(n_sim, n_steps, params):
    """
    Simulate wind using a Generalized Hyperbolic (GHYP) distribution model.
    """
    # Placeholder: Replace with actual GHYP simulation code
    # For demonstration, we'll simulate wind speed using a skewed t-distribution
    # GHYP can capture heavy tails and skewness
    df = params['wind_df']
    loc = params['wind_loc']
    scale = params['wind_scale']
    skew = params['wind_skew']  # Skewness parameter

    # Using skewed t-distribution as an approximation for GHYP
    wind = skew * t.rvs(df, loc=loc, scale=scale, size=(n_sim, n_steps))
    return wind

# Define simulation parameters
weather_params = {
    'rain_shape': 2.0,     # Shape parameter for Gamma distribution
    'rain_scale': 1.0,     # Scale parameter for Gamma distribution
    'theta': 0.7,          # Speed of reversion for temperature
    'mu': 15.0,            # Long-term mean for temperature
    'sigma': 5.0,          # Volatility for temperature
    'T0': 15.0,            # Initial temperature
    'wind_df': 5,          # Degrees of freedom for wind (t-distribution)
    'wind_loc': 10.0,      # Location parameter for wind
    'wind_scale': 3.0,     # Scale parameter for wind
    'wind_skew': 1.0       # Skewness for wind
}

# Simulation settings
n_sim = 10000       # Number of simulations
n_steps = 365       # Number of daily steps (1 year)
dt = 1/365          # Time step size

# Simulate weather variables
print("Simulating rainfall data...")
rainfall = simulate_rainfall(n_sim, n_steps, weather_params)

print("Simulating temperature data...")
temperature = simulate_temperature(n_sim, n_steps, dt, weather_params)

print("Simulating wind data...")
wind = simulate_wind(n_sim, n_steps, weather_params)

# ----------------------------
# 2. Model Joint Behavior
# ----------------------------

def fit_copula(rainfall, temperature, wind):
    """
    Fit a dynamic copula model to the three weather variables.
    For simplicity, we'll use Gaussian copula with polynomial regression dependencies.
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from scipy.stats import norm

    # Flatten the data for regression
    X = np.column_stack((rainfall.flatten(), temperature.flatten(), wind.flatten()))
    
    # Apply polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Fit a linear regression model to model dependencies
    model = LinearRegression()
    model.fit(X_poly, X_poly)  # Self-predicting for demonstration
    
    # Extract residuals
    predictions = model.predict(X_poly)
    residuals = X_poly - predictions
    
    # Standardize residuals
    residuals_std = (residuals - residuals.mean(axis=0)) / residuals.std(axis=0)
    
    # Apply Gaussian copula
    copula = norm.cdf(residuals_std)
    
    return copula

# Fit copula (simplified for demonstration)
print("Fitting copula model...")
copula = fit_copula(rainfall, temperature, wind)

# Note: need more complex - dynamic copulas, and properly handle time dependencies.

# ----------------------------
# 3. Define Basket Derivative Payoff
# ----------------------------

def basket_derivative_payoff(rainfall, temperature, wind, weights):
    """
    Define the payoff of a basket derivative combining rainfall, temperature, and wind.
    Payoff could be, for example, a weighted sum of option payoffs on each variable.
    
    Parameters:
    - rainfall: Simulated rainfall paths
    - temperature: Simulated temperature paths
    - wind: Simulated wind paths
    - weights: Dictionary with weights for each variable
    
    Returns:
    - payoff: Payoff matrix (n_sim,)
    """
    # Define strike prices for each component
    K_rain = 10.0
    K_temp = 15.0
    K_wind = 12.0

    # Example payoff: sum of individual call option payoffs
    payoff_rain = np.maximum(rainfall[:, -1] - K_rain, 0)
    payoff_temp = np.maximum(temperature[:, -1] - K_temp, 0)
    payoff_wind = np.maximum(wind[:, -1] - K_wind, 0)

    # Weighted sum
    payoff = weights['rain'] * payoff_rain + weights['temp'] * payoff_temp + weights['wind'] * payoff_wind

    return payoff

# Define weights for the basket
basket_weights = {
    'rain': 1.0,
    'temp': 1.0,
    'wind': 1.0
}

# Compute payoff
print("Computing basket derivative payoffs...")
payoff = basket_derivative_payoff(rainfall, temperature, wind, basket_weights)

# ----------------------------
# 4. Define Utility Function
# ----------------------------

def power_utility(x, gamma):
    """
    Power utility function.
    """
    # Ensure x is positive
    x = np.maximum(x, 1e-8)
    return (x ** (1 - gamma)) / (1 - gamma)

def certainty_equivalent(expected_utility, gamma):
    """
    Compute the certainty equivalent for power utility.
    """
    return (expected_utility * (1 - gamma)) ** (1 / (1 - gamma))

# ----------------------------
# 5. Calculate Indifference Price Bounds
# ----------------------------

def compute_indifference_price_bounds(payoff, gamma, a, K):
    """
    Compute the lower and upper bounds for the indifference price under power utility.
    
    Parameters:
    - payoff: Array of option payoffs (n_sim,)
    - gamma: Risk aversion parameter
    - a: Coefficient in value function formula (from the reference)
    - K: Strike price or relevant constant
    
    Returns:
    - pi_low: Lower bound of indifference price
    - pi_up: Upper bound of indifference price
    """
    # Lower Bound: Certainty Equivalent
    U_payoff = power_utility(payoff, gamma)
    EU_payoff = np.mean(U_payoff)
    CE_payoff = certainty_equivalent(EU_payoff, gamma)
    
    # Since in indifference pricing, the price should satisfy:
    # U(W) = U(W - pi + Payoff)
    # Here, for bounds, we consider:
    # pi_low <= Indiff Price <= pi_up
    
    # Lower bound could be CE_payoff - current wealth (assuming wealth=0)
    pi_low = CE_payoff
    
    # Upper bound: Maximum possible payoff
    pi_up = np.max(payoff)
    
    return pi_low, pi_up

# Example parameters for indifference pricing
gamma = 0.5  # Risk aversion
a = 0.5      # Coefficient (from the reference)
K = 10.0     # Strike price component

# Compute indifference price bounds
print("Calculating indifference price bounds...")
pi_low, pi_up = compute_indifference_price_bounds(payoff, gamma, a, K)
print(f"Indifference Price Lower Bound: {pi_low:.4f}")
print(f"Indifference Price Upper Bound: {pi_up:.4f}")

# ----------------------------
# 6. Sensitivity Analysis
# ----------------------------

def sensitivity_analysis(payoff, gamma_values, rho_values, a, K):
    """
    Perform sensitivity analysis on indifference price bounds across different
    risk aversion and correlation levels.
    
    Parameters:
    - payoff: Array of option payoffs (n_sim,)
    - gamma_values: Array of gamma (risk aversion) values
    - rho_values: Array of rho (correlation) values
    - a: Coefficient in value function formula
    - K: Strike price or relevant constant
    
    Returns:
    - results_df: DataFrame containing the bounds for each combination
    """
    results = []
    for gamma in gamma_values:
        for rho in rho_values:
            # Adjust the correlation if necessary
            # In this simplified example, rho affects the dependency structure
            # Here, we keep it as a placeholder since copula fitting is simplified
            
            # Recompute payoff if correlation affects it
            # For demonstration, assume payoff remains the same
            # In practice, correlation would affect the joint distribution and payoffs
            
            # Compute bounds
            pi_low, pi_up = compute_indifference_price_bounds(payoff, gamma, a, K)
            
            results.append({
                'Risk Aversion (gamma)': gamma,
                'Correlation (rho)': rho,
                'Indifference Price Lower Bound': pi_low,
                'Indifference Price Upper Bound': pi_up
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Define ranges for gamma and rho
gamma_values = np.linspace(0.1, 2.0, 10)  # 10 values from 0.1 to 2.0
rho_values = np.linspace(-0.9, 0.9, 10)   # 10 values from -0.9 to 0.9

# Perform sensitivity analysis
print("Performing sensitivity analysis...")
results_df = sensitivity_analysis(payoff, gamma_values, rho_values, a, K)

# Save results to CSV
results_df.to_csv(os.path.join(save_dir, 'indifference_price_bounds.csv'), index=False)
print(f"Indifference price bounds saved to {os.path.join(save_dir, 'indifference_price_bounds.csv')}")


############################# Results and Visualization ################################

# ----------------------------
# 7. Advanced Visualizations
# ----------------------------

def plot_heatmap(data, x, y, z, title, xlabel, ylabel, filename, cmap='viridis'):
    """
    Plot a heatmap using seaborn.
    """
    
    pivot_table = data.pivot(index=y, columns=x, values=z)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()

def plot_line_chart(data, x, y, hue, title, xlabel, ylabel, filename):
    """
    Plot a line chart using seaborn.
    """
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x=x, y=y, hue=hue, marker='o')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title=hue)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()

def plot_3d_surface(data, x, y, z, title, xlabel, ylabel, zlabel, filename):
    """
    Plot a 3D surface.
    """
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import griddata

    # Prepare grid data
    grid_x, grid_y = np.mgrid[data[x].min():data[x].max():100j, data[y].min():data[y].max():100j]
    grid_z = griddata((data[x], data[y]), data[z], (grid_x, grid_y), method='cubic')
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()

# Heatmap for Lower Bounds
print("Plotting heatmap for lower bounds...")
plot_heatmap(
    data=results_df,
    x='Correlation (rho)',
    y='Risk Aversion (gamma)',
    z='Indifference Price Lower Bound',
    title='Lower Bound of Indifference Price (Power Utility)',
    xlabel='Correlation (rho)',
    ylabel='Risk Aversion (gamma)',
    filename='heatmap_lower_bound.png',
    cmap='YlGnBu'
)

# Heatmap for Upper Bounds
print("Plotting heatmap for upper bounds...")
plot_heatmap(
    data=results_df,
    x='Correlation (rho)',
    y='Risk Aversion (gamma)',
    z='Indifference Price Upper Bound',
    title='Upper Bound of Indifference Price (Power Utility)',
    xlabel='Correlation (rho)',
    ylabel='Risk Aversion (gamma)',
    filename='heatmap_upper_bound.png',
    cmap='YlOrRd'
)

# Line Chart: Indifference Price Bounds vs Correlation for Different Gamma
print("Plotting line chart for bounds vs correlation...")
# For each gamma, plot bounds across rho
plt.figure(figsize=(12, 8))
for gamma in gamma_values:
    subset = results_df[results_df['Risk Aversion (gamma)'] == gamma]
    plt.plot(subset['Correlation (rho)'], subset['Indifference Price Lower Bound'], 
             marker='o', linestyle='--', label=f'gamma={gamma:.1f} Lower')
    plt.plot(subset['Correlation (rho)'], subset['Indifference Price Upper Bound'], 
             marker='x', linestyle='-', label=f'gamma={gamma:.1f} Upper')
plt.title('Indifference Price Bounds vs Correlation for Different Risk Aversion Levels', fontsize=16)
plt.xlabel('Correlation (rho)', fontsize=12)
plt.ylabel('Indifference Price Bound', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'indifference_price_bounds_vs_correlation.png'))
plt.show()

# Line Chart: Indifference Price Bounds vs Risk Aversion for Different Rho
print("Plotting line chart for bounds vs risk aversion...")
# For each rho, plot bounds across gamma
plt.figure(figsize=(12, 8))
for rho in rho_values:
    subset = results_df[results_df['Correlation (rho)'] == rho]
    plt.plot(subset['Risk Aversion (gamma)'], subset['Indifference Price Lower Bound'], 
             marker='o', linestyle='--', label=f'rho={rho:.1f} Lower')
    plt.plot(subset['Risk Aversion (gamma)'], subset['Indifference Price Upper Bound'], 
             marker='x', linestyle='-', label=f'rho={rho:.1f} Upper')
plt.title('Indifference Price Bounds vs Risk Aversion for Different Correlation Levels', fontsize=16)
plt.xlabel('Risk Aversion (gamma)', fontsize=12)
plt.ylabel('Indifference Price Bound', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'indifference_price_bounds_vs_risk_aversion.png'))
plt.show()

# 3D Surface Plot for Lower Bound
print("Plotting 3D surface plot for lower bounds...")
plot_3d_surface(
    data=results_df,
    x='Risk Aversion (gamma)',
    y='Correlation (rho)',
    z='Indifference Price Lower Bound',
    title='Lower Bound of Indifference Price (Power Utility)',
    xlabel='Risk Aversion (gamma)',
    ylabel='Correlation (rho)',
    zlabel='Lower Bound Price',
    filename='3D_lower_bound.png'
)

# 3D Surface Plot for Upper Bound
print("Plotting 3D surface plot for upper bounds...")
plot_3d_surface(
    data=results_df,
    x='Risk Aversion (gamma)',
    y='Correlation (rho)',
    z='Indifference Price Upper Bound',
    title='Upper Bound of Indifference Price (Power Utility)',
    xlabel='Risk Aversion (gamma)',
    ylabel='Correlation (rho)',
    zlabel='Upper Bound Price',
    filename='3D_upper_bound.png'
)

# ----------------------------
# 8. Tabulate Results
# ----------------------------

def save_summary_table(data, filename):
    """
    Save summary table to CSV and display it.
    """
    data.to_csv(os.path.join(save_dir, filename), index=False)
    print(f"Summary table saved to {os.path.join(save_dir, filename)}")
    print(data.head())

# Save and display summary tables
print("Saving and displaying summary tables...")
save_summary_table(results_df, 'indifference_price_bounds_summary.csv')

# ----------------------------
# 9. Simulate HJB PDE (Simplified)
# ----------------------------

def simulate_hjb_pde(payoff, gamma, a, K, n_steps, dt):
    """
    Simulate the HJB PDE using a simplified finite difference method.
    For demonstration purposes, we'll use a 1D PDE. Extending to multidimensional requires
    more sophisticated numerical schemes.
    
    Parameters:
    - payoff: Array of option payoffs (n_sim,)
    - gamma: Risk aversion parameter
    - a: Coefficient in value function formula
    - K: Strike price or relevant constant
    - n_steps: Number of time steps
    - dt: Time step size
    
    Returns:
    - value_function: Approximated value function at each step
    """
    # Placeholder: Implement a simplified HJB PDE solver
    # Implementing a full multidimensional PDE solver is beyond this scope
    # For demonstration, we return a mock value function
    
    value_function = np.linspace(0, 1, n_steps)
    return value_function

# Simulate HJB PDE (simplified)
print("Simulating HJB PDE...")
value_function = simulate_hjb_pde(payoff, gamma, a, K, n_steps, dt)

# ----------------------------
# 10. Final Indifference Price from PDE
# ----------------------------

def compute_indifference_price_from_pde(value_function, gamma):
    """
    Compute the indifference price from the PDE solution.
    For demonstration, we'll use the final value of the value function.
    
    Parameters:
    - value_function: Array representing the value function over time
    - gamma: Risk aversion parameter
    
    Returns:
    - indiff_price: Indifference price derived from the PDE
    """
    # Placeholder: Actual computation would involve integrating the PDE solution
    indiff_price = value_function[-1] * gamma  # Simplified formula
    return indiff_price

# Compute indifference price from PDE
indiff_price_pde = compute_indifference_price_from_pde(value_function, gamma)
print(f"Indifference Price from PDE: {indiff_price_pde:.4f}")

# ----------------------------
# 11. Final Visualization and Summary
# ----------------------------

# Plot Value Function
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, T, n_steps), value_function, label='Value Function')
plt.title('Value Function over Time')
plt.xlabel('Time (years)')
plt.ylabel('Value Function V(t)')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'value_function.png'))
plt.show()

# Summary Table including PDE-based Indifference Price
summary_pde = pd.DataFrame({
    'Gamma': [gamma],
    'Indifference Price from PDE': [indiff_price_pde]
})

summary_pde.to_csv(os.path.join(save_dir, 'indifference_price_pde.csv'), index=False)
print("\nPDE-based Indifference Price Summary:")
print(summary_pde)

# ----------------------------
# 12. Combined Indifference Price Summary
# ----------------------------

# Merge bounds and PDE-based price
combined_summary = results_df.copy()
combined_summary['Indifference Price from PDE'] = indiff_price_pde

# Save combined summary
combined_summary.to_csv(os.path.join(save_dir, 'combined_indifference_price_summary.csv'), index=False)
print(f"\nCombined indifference price summary saved to {os.path.join(save_dir, 'combined_indifference_price_summary.csv')}")

# Display combined summary
print("\nCombined Indifference Price Summary:")
print(combined_summary.head())

