#buyer and seller 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import pandas as pd
import os

# Directory to save the results
save_dir = '/content/drive/MyDrive/wd/'
os.makedirs(save_dir, exist_ok=True)

# Parameters
T = 1  # in years
n_sim = 10000  # Number of Monte Carlo simulations
n_steps = 365  # Time steps
dt = T / n_steps  # Time step size
alpha = 0.5  # Risk aversion
a = 0.5  # Coefficient in the value function formula

# Parameters for S_t and Y_t
mu_Y = 0.1  # Drift of non-traded asset Y
sigma_Y = 0.1  # Volatility of Y
rho = 0.5  # Correlation between S and Y
K = 20  # Strike price for call option on Y
Y0 = 20  # Initial value of Y

# OLS-based dependence for drift of traded asset S
b = 0.4
epsilon = 0.1
mu_S = lambda mu_Y: a + b * mu_Y + epsilon  # Drift of the traded asset S
sigma_S = 0.2  # Volatility of traded asset S

# Function to simulate S and Y paths
def simulate_paths(n_sim, n_steps, dt, mu_Y, sigma_Y, rho):
    S_paths = np.zeros((n_sim, n_steps))
    Y_paths = np.zeros((n_sim, n_steps))

    S_paths[:, 0] = 100  # Initial stock price
    Y_paths[:, 0] = Y0  # Initial Y

    for t in range(1, n_steps):
        dW_Y = np.random.normal(0, np.sqrt(dt), n_sim)
        dW_S = rho * dW_Y + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), n_sim)
        
        Y_paths[:, t] = Y_paths[:, t - 1] + mu_Y * dt + sigma_Y * dW_Y
        mu_S_t = mu_S(mu_Y)
        S_paths[:, t] = S_paths[:, t - 1] * np.exp((mu_S_t - 0.5 * sigma_S**2) * dt + sigma_S * dW_S)

    return S_paths, Y_paths

# Value function integral calculation
def compute_value_function(S_paths, Y_paths, alpha, a, mu_Y, sigma_Y):
    # Initialize the integral for each Monte Carlo path
    integral = np.zeros(n_sim)

    # Time integral: ∫_t^T [a * g(s, Y_s) + (1/2) * (mu^2(Y_s) / sigma^2(Y_s))] ds
    for t in range(n_steps):
        # g(s, Y_s): Call option payoff at each step, but integrated over time, not averaged
        payoff_integrand = a * np.maximum(Y_paths[:, t] - K, 0)
        risk_integrand = 0.5 * (mu_S(mu_Y)**2) / (sigma_S**2)

        # Accumulate the integral over time
        integral += (payoff_integrand + risk_integrand) * dt

    # After the integral has been computed, apply the exponential utility for each path
    path_utilities = np.exp(-integral)

    # Compute the expectation over Monte Carlo paths (averaging at the final time step)
    value_function = -np.exp(-alpha * 100) * np.mean(path_utilities)

    return value_function


# Function to compute indifference price
def compute_indifference_price(S_paths, Y_paths, alpha, a, mu_Y, sigma_Y):
    # Compute the integrals for both the numerator and denominator
    integral_no_option = np.zeros(n_sim)
    integral_with_option = np.zeros(n_sim)

    # Time integral for each Monte Carlo path
    for t in range(n_steps):
        # No option integral: involves just the risk component
        risk_integrand = 0.5 * (mu_S(mu_Y)**2) / (sigma_S**2)
        integral_no_option += risk_integrand * dt

        # With option integral: involves both the payoff and risk
        payoff_integrand = a * np.maximum(Y_paths[:, t] - K, 0)
        integral_with_option += (payoff_integrand + risk_integrand) * dt

    # Compute the exponential terms for both cases
    utility_no_option = np.exp(-integral_no_option)
    utility_with_option = np.exp(-integral_with_option)

    # Calculate the indifference price using the log ratio of expectations
    expected_no_option = np.mean(utility_no_option)
    expected_with_option = np.mean(utility_with_option)

    # (option price V is obtained as the difference between certainity equivalents with and without n derivative claims)
    indifference_price = (1 / alpha) * np.log(expected_no_option / expected_with_option)

    return indifference_price

# Simulate paths
S_paths, Y_paths = simulate_paths(n_sim, n_steps, dt, mu_Y, sigma_Y, rho)

# Compute indifference price
indifference_price = compute_indifference_price(S_paths, Y_paths, alpha, a, mu_Y, sigma_Y)

print(f"Indifference Price: {indifference_price}")


# Simulate paths
S_paths, Y_paths = simulate_paths(n_sim, n_steps, dt, mu_Y, sigma_Y, rho)

# Compute value function
value_function = compute_value_function(S_paths, Y_paths, alpha, a, mu_Y, sigma_Y)

# Radon-Nikodym derivative (change of measure from P to Q)
W_T_S = np.sum(np.random.normal(0, np.sqrt(dt), size=(n_sim, n_steps)), axis=1)
radon_nikodym = np.exp(-mu_S(mu_Y) / sigma_S * W_T_S - (mu_S(mu_Y)**2 * T) / (2 * sigma_S**2))

# Indifference price
indifference_price = compute_indifference_price(S_paths, Y_paths, alpha, a, mu_Y, sigma_Y)

############################################## Sensitivity Analysis - against Correlation and Risk Averison assumptions ###################################################

# 3D Plot: Value function vs correlation and time
correlation_values = np.linspace(-0.9, 0.9, 10)
time_values = np.linspace(0, T, n_steps)

indifference_prices_corr = []

for rho in correlation_values:
    S_paths, Y_paths = simulate_paths(n_sim, n_steps, dt, mu_Y, sigma_Y, rho)
    indifference_price = compute_indifference_price(S_paths, Y_paths, alpha, radon_nikodym, mu_Y, sigma_Y)
    indifference_prices_corr.append(indifference_price)

# Use cubic spline interpolation for smooth curves
cs = CubicSpline(correlation_values, indifference_prices_corr)
smooth_correlation_values = np.linspace(-0.9, 0.9, 100)
indifference_prices_smooth = cs(smooth_correlation_values)

# Plot: Indifference price vs correlation (with spline)
plt.figure(figsize=(8, 6))
plt.plot(correlation_values, indifference_prices_corr, 'o', label='Algorithm', color='blue')
plt.plot(smooth_correlation_values, indifference_prices_smooth, '-', label='Cubic Spline Fit', color='red')

# Apply formatting to match style in the provided image
plt.xlabel('Correlation $\\rho$', fontsize=12)
plt.ylabel('Indifference Price $\\pi^{cont}$', fontsize=12)
plt.title('Indifference Price vs Correlation', fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(os.path.join(save_dir, 'indifference_price_vs_correlation.png'))
plt.show()

# Risk aversion sensitivity
alpha_values = np.linspace(0.1, 2.0, 10)
indifference_prices_alpha = [] 

for alpha_val in alpha_values:
    indifference_price = compute_indifference_price(S_paths, Y_paths, alpha_val, radon_nikodym, mu_Y, sigma_Y)
    indifference_prices_alpha.append(indifference_price)

# Polynomial fit for risk aversion
polynomial_fit = np.poly1d(np.polyfit(alpha_values, indifference_prices_alpha, 3))
smooth_alpha_values = np.linspace(0.1, 2.0, 100)
indifference_prices_smooth_alpha = polynomial_fit(smooth_alpha_values)

# Plot: Indifference price vs risk aversion
plt.figure(figsize=(8, 6))
plt.plot(alpha_values, indifference_prices_alpha, 'o', label='Algorithm', color='blue')
plt.plot(smooth_alpha_values, indifference_prices_smooth_alpha, '-', label='Polynomial Fit', color='red')

# Apply formatting to match style in the provided image
plt.xlabel('Risk Aversion $\\alpha$', fontsize=12)
plt.ylabel('Indifference Price $\\pi^{cont}$', fontsize=12)
plt.title('Indifference Price vs Risk Aversion', fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(os.path.join(save_dir, 'indifference_price_vs_risk_aversion.png'))
plt.show()

# Summary Table for Results
data_summary = {
    'Correlation (rho)': correlation_values,
    'Indifference Price (Correlation)': indifference_prices_corr,
    'Risk Aversion (alpha)': alpha_values,
    'Indifference Price (Risk Aversion)': indifference_prices_alpha
}
df_summary = pd.DataFrame(data_summary)

# Display results summary
import ace_tools_open as tools; tools.display_dataframe_to_user(name="Indifference Price Summary", dataframe=df_summary)

# Revised function to compute value function over varying time horizons
def compute_value_function_over_time(S_paths, Y_paths, alpha, a, mu_Y, sigma_Y, t_steps):
    n_sim = S_paths.shape[0]
    integral = np.zeros(n_sim)

    # Compute the integral only over the available time steps (t_steps)
    for t in range(t_steps):
        payoff_integrand = a * np.maximum(Y_paths[:, t] - K, 0)
        risk_integrand = 0.5 * (mu_S(mu_Y)**2) / (sigma_S**2)
        integral += (payoff_integrand + risk_integrand) * dt

    # After the integral has been computed, apply the exponential utility for each path
    path_utilities = np.exp(-integral)

    # Compute the expectation over Monte Carlo paths (averaging at the final time step)
    value_function = -np.exp(-alpha * 100) * np.mean(path_utilities)

    return value_function

# Now adjust the value function matrix calculation
value_function_matrix = np.zeros((len(correlation_values), len(time_values)))

for i, rho in enumerate(correlation_values):
    S_paths, Y_paths = simulate_paths(n_sim, n_steps, dt, mu_Y, sigma_Y, rho)
    for j, t in enumerate(time_values):
        # For each time `t_j`, compute the value function up to that time horizon
        t_steps = j + 1  # Use time steps up to t_j
        value_function_matrix[i, j] = compute_value_function_over_time(S_paths, Y_paths, alpha, a, mu_Y, sigma_Y, t_steps)

# 3D Plot of value function against correlation and time
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(correlation_values, time_values)
Z = value_function_matrix.T

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('Correlation $\\rho$', fontsize=12)
ax.set_ylabel('Time $t$', fontsize=12)
ax.set_zlabel('Value Function $V(t, \\rho)$', fontsize=12)
ax.set_title('3D Plot of Value Function vs Correlation and Time', fontsize=14)
plt.savefig(os.path.join(save_dir, 'value_function_3d_plot.png'))
plt.show()

# 2D plots for sensitivity analysis

# Cubic spline for indifference price vs correlation
cs = CubicSpline(correlation_values, np.mean(value_function_matrix, axis=1))
smooth_correlation_values = np.linspace(-0.9, 0.9, 100)
indifference_prices_smooth = cs(smooth_correlation_values)

plt.figure(figsize=(8, 6))
plt.plot(correlation_values, np.mean(value_function_matrix, axis=1), 'o', label='Simulated Data', color='blue')
plt.plot(smooth_correlation_values, indifference_prices_smooth, '-', label='Cubic Spline Fit', color='red')
plt.xlabel('Correlation $\\rho$', fontsize=12)
plt.ylabel('Value Function $V(t, \\rho)$', fontsize=12)
plt.title('Value Function vs Correlation', fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(os.path.join(save_dir, 'value_function_vs_correlation.png'))
plt.show()

# Summary Table for Results
data_summary = {
    'Correlation (rho)': correlation_values,
    'Average Value Function': np.mean(value_function_matrix, axis=1)
}
df_summary = pd.DataFrame(data_summary)

# Display results summary
import ace_tools_open as tools; tools.display_dataframe_to_user(name="Value Function Summary", dataframe=df_summary)

# Save the summary table to a CSV file
df_summary.to_csv(os.path.join(save_dir, 'value_function_summary.csv'))

