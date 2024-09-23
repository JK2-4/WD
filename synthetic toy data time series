import numpy as np
import matplotlib.pyplot as plt

# Time settings
T = 1.0  # Time horizon in years
N = 365  # Daily data points
dt = T / N
t = np.linspace(0, T, N)

# Mean-reversion parameters for temperature
kappa_T = 0.3  # Speed of mean reversion
theta_T = 15 + 10 * np.sin(2 * np.pi * t)  # Seasonal mean
sigma_T = 3  # Volatility of temperature

# Wind speed parameters (GHYP/NIG)
mu_W = 8  # Mean wind speed
sigma_W = 1.5  # Volatility
kappa_W = 0.2  # Mean reversion speed
jumps_W = np.random.poisson(0.02, N)  # Simulating jumps in wind speed

# Humidity parameters
kappa_H = 0.4
theta_H = 70 + 10 * np.cos(2 * np.pi * t)
sigma_H = 5

# Initialize arrays
T_series = np.zeros(N)
W_series = np.zeros(N)
H_series = np.zeros(N)
T_series[0] = 15
W_series[0] = mu_W
H_series[0] = 70

# Generate synthetic data using SDEs
for i in range(1, N):
    T_series[i] = T_series[i-1] + kappa_T * (theta_T[i-1] - T_series[i-1]) * dt + sigma_T * np.sqrt(dt) * np.random.normal()
    W_series[i] = W_series[i-1] + kappa_W * (mu_W - W_series[i-1]) * dt + sigma_W * np.sqrt(dt) * np.random.normal() + jumps_W[i]
    H_series[i] = H_series[i-1] + kappa_H * (theta_H[i-1] - H_series[i-1]) * dt + sigma_H * np.sqrt(dt) * np.random.normal()

# Plot the generated data
plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.plot(t, T_series, label="Temperature (°C)")
plt.legend()
plt.subplot(312)
plt.plot(t, W_series, label="Wind Speed (m/s)", color='orange')
plt.legend()
plt.subplot(313)
plt.plot(t, H_series, label="Humidity (%)", color='green')
plt.legend()
plt.show()
