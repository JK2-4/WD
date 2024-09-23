################PDE numerical scheme##############################################
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1/3  # Maturity (4 months)
r = 0.05  # Risk-free rate
gamma = 0.1  # Risk aversion coefficient (adjusted as per your request)

# Traded asset parameters (S_t)
mu_S = 0.1     # Drift of the traded asset
sigma_S = 0.2  # Volatility of the traded asset

# Non-traded asset parameters (Y_t)
kappa = 1.0    # Mean-reversion speed of the temperature
mu_Y = 20.0    # Long-term mean temperature
sigma_Y = 0.1  # Volatility of the temperature

# Initial temperature Y0
Y0 = 20.0  # Set to the long-term mean for simplicity

# Set strike price K equal to Y0
K = Y0

# Multiplier M (adjust based on variance criteria or set as needed)
M = 100.0  # For demonstration, we'll set M = 100

# Correlation between S_t and Y_t
rho = 0.5  # We'll vary this later in sensitivity analysis

# Grid parameters for finite difference method
W_min, W_max, dW = -50.0, 50.0, 1.0  # Wealth grid
Y_min, Y_max, dY = 10.0, 30.0, 0.5    # Temperature grid
dt = 0.005  # Time step size

# Create grids
W_grid = np.arange(W_min, W_max + dW, dW)  # Wealth grid points
Y_grid = np.arange(Y_min, Y_max + dY, dY)  # Temperature grid points
time_grid = np.arange(0, T + dt, dt)       # Time grid points

N_W = len(W_grid)  # Number of wealth grid points
N_Y = len(Y_grid)  # Number of temperature grid points
N_t = len(time_grid)  # Number of time steps

# Initialize the distorted value function Phi
Phi = np.zeros((N_W, N_Y, N_t))

# Define the payoff function
def payoff(Y):
    """
    Payoff function of the call option on temperature with multiplier M.
    Payoff = M * max(Y_T - K, 0)
    """
    return M * np.maximum(Y - K, 0)

# Define the terminal condition for Phi at t = T
def terminal_condition(W, Y):
    """
    Terminal condition based on the utility function.
    Phi(T, W, Y) = -exp(-gamma * (W + payoff(Y)))
    """
    H_Y = payoff(Y)
    return -np.exp(-gamma * (W + H_Y))

# Set terminal condition at t = T for all grid points
for i in range(N_W):
    for j in range(N_Y):
        Phi[i, j, -1] = terminal_condition(W_grid[i], Y_grid[j])

# Precompute constants used in the PDE coefficients
lambda_S = (mu_S - r) / sigma_S  # Market price of risk for the traded asset
alpha = r - (lambda_S ** 2) / (2 * gamma)  # Drift adjustment in the PDE

# Define the coefficients for the finite difference scheme
def coefficients():
    """
    Compute the coefficients for the finite difference scheme corresponding to the PDE terms.
    """
    # Coefficients for second derivatives
    a_WW = 0.5 * sigma_S**2
    a_YY = 0.5 * sigma_Y**2
    a_WY = rho * sigma_S * sigma_Y

    # Coefficients for first derivatives
    b_W = r * W_grid - (lambda_S / gamma) * sigma_S**2
    b_Y = kappa * (mu_Y - Y_grid)

    return a_WW, a_YY, a_WY, b_W, b_Y

a_WW, a_YY, a_WY, b_W, b_Y = coefficients()

# Main time-stepping loop (backward in time)
for n in reversed(range(N_t - 1)):
    # Copy the next time step's Phi for calculations
    Phi_next = Phi[:, :, n + 1].copy()
    Phi_current = Phi[:, :, n].copy()

    # Loop over the wealth grid (excluding boundaries)
    for i in range(1, N_W - 1):
        # Loop over the temperature grid (excluding boundaries)
        for j in range(1, N_Y - 1):
            # Compute second derivatives using central differences
            Phi_WW = (Phi_next[i + 1, j] - 2 * Phi_next[i, j] + Phi_next[i - 1, j]) / (dW**2)
            Phi_YY = (Phi_next[i, j + 1] - 2 * Phi_next[i, j] + Phi_next[i, j - 1]) / (dY**2)

            # Compute mixed derivative
            Phi_WY = (Phi_next[i + 1, j + 1] - Phi_next[i + 1, j - 1] - Phi_next[i - 1, j + 1] + Phi_next[i - 1, j - 1]) / (4 * dW * dY)

            # Compute first derivatives
            Phi_W = (Phi_next[i + 1, j] - Phi_next[i - 1, j]) / (2 * dW)
            Phi_Y = (Phi_next[i, j + 1] - Phi_next[i, j - 1]) / (2 * dY)

            # Update Phi using the PDE (linearized HJB equation)
            Phi_current[i, j] = Phi_next[i, j] + dt * (
                -alpha * Phi_W  # Drift term
                + a_WW * Phi_WW  # Diffusion term for wealth
                + a_YY * Phi_YY  # Diffusion term for temperature
                + a_WY * Phi_WY  # Mixed term due to correlation
                - b_W[i] * Phi_W  # Wealth gradient term
                - b_Y[j] * Phi_Y  # Temperature gradient term
            )

    # Apply Neumann boundary conditions (zero gradient at boundaries)
    Phi_current[0, :] = Phi_current[1, :]  # At W_min
    Phi_current[-1, :] = Phi_current[-2, :]  # At W_max
    Phi_current[:, 0] = Phi_current[:, 1]  # At Y_min
    Phi_current[:, -1] = Phi_current[:, -2]  # At Y_max

    # Update Phi for the current time step
    Phi[:, :, n] = Phi_current

# Compute the value function V from the distorted function Phi
# Using the distortion transformation: V = - (1/gamma) * log(-Phi)
V = -np.log(-Phi) / gamma

# Extract the value function at initial wealth W0 and temperature Y0
W0 = 0.0  # Initial wealth
i_W0 = np.argmin(np.abs(W_grid - W0))  # Index for W0
j_Y0 = np.argmin(np.abs(Y_grid - Y0))  # Index for Y0

V_with_option = V[i_W0, j_Y0, 0]  # Value with the option

# Repeat the process without the option to get V_without_option
# Update terminal condition without the option payoff
for i in range(N_W):
    for j in range(N_Y):
        Phi[i, j, -1] = -np.exp(-gamma * W_grid[i])  # No payoff from the option

# Time-stepping loop without the option
for n in reversed(range(N_t - 1)):
    Phi_next = Phi[:, :, n + 1].copy()
    Phi_current = Phi[:, :, n].copy()

    for i in range(1, N_W - 1):
        for j in range(1, N_Y - 1):
            # Same calculations as before
            Phi_WW = (Phi_next[i + 1, j] - 2 * Phi_next[i, j] + Phi_next[i - 1, j]) / (dW**2)
            Phi_YY = (Phi_next[i, j + 1] - 2 * Phi_next[i, j] + Phi_next[i, j - 1]) / (dY**2)
            Phi_WY = (Phi_next[i + 1, j + 1] - Phi_next[i + 1, j - 1] - Phi_next[i - 1, j + 1] + Phi_next[i - 1, j - 1]) / (4 * dW * dY)
            Phi_W = (Phi_next[i + 1, j] - Phi_next[i - 1, j]) / (2 * dW)
            Phi_Y = (Phi_next[i, j + 1] - Phi_next[i, j - 1]) / (2 * dY)

            # Update Phi using the PDE (without the option payoff)
            Phi_current[i, j] = Phi_next[i, j] + dt * (
                -alpha * Phi_W
                + a_WW * Phi_WW
                + a_YY * Phi_YY
                + a_WY * Phi_WY
                - b_W[i] * Phi_W
                - b_Y[j] * Phi_Y
            )

    # Apply boundary conditions
    Phi_current[0, :] = Phi_current[1, :]
    Phi_current[-1, :] = Phi_current[-2, :]
    Phi_current[:, 0] = Phi_current[:, 1]
    Phi_current[:, -1] = Phi_current[:, -2]

    # Update Phi
    Phi[:, :, n] = Phi_current

# Compute the value function V without the option
V_no_option = -np.log(-Phi) / gamma

V_without_option = V_no_option[i_W0, j_Y0, 0]  # Value without the option

# Compute the indifference price
indifference_price = V_without_option - V_with_option

print(f"Indifference Price: {indifference_price:.4f}")

# Visualization: Plot the value function V at initial time t = 0
plt.figure(figsize=(8, 6))
plt.contourf(Y_grid, W_grid, V[:, :, 0], levels=50, cmap='viridis')
plt.colorbar(label='Value Function V')
plt.xlabel('Temperature Y')
plt.ylabel('Wealth W')
plt.title('Value Function V at t=0')
plt.show()

# Sensitivity analysis for different risk aversion values
gamma_values = [0.05, 0.1, 0.5, 1.0]  # Different gamma values
indifference_prices_gamma = []

for gamma in gamma_values:
    # Update alpha with new gamma
    alpha = r - (lambda_S ** 2) / (2 * gamma)

    # Update terminal condition with new gamma
    for i in range(N_W):
        for j in range(N_Y):
            Phi[i, j, -1] = -np.exp(-gamma * (W_grid[i] + payoff(Y_grid[j])))

    # Time-stepping loop with updated gamma
    for n in reversed(range(N_t - 1)):
        Phi_next = Phi[:, :, n + 1].copy()
        Phi_current = Phi[:, :, n].copy()

        for i in range(1, N_W - 1):
            for j in range(1, N_Y - 1):
                Phi_WW = (Phi_next[i + 1, j] - 2 * Phi_next[i, j] + Phi_next[i - 1, j]) / (dW**2)
                Phi_YY = (Phi_next[i, j + 1] - 2 * Phi_next[i, j] + Phi_next[i, j - 1]) / (dY**2)
                Phi_WY = (Phi_next[i + 1, j + 1] - Phi_next[i + 1, j - 1] - Phi_next[i - 1, j + 1] + Phi_next[i - 1, j - 1]) / (4 * dW * dY)
                Phi_W = (Phi_next[i + 1, j] - Phi_next[i - 1, j]) / (2 * dW)
                Phi_Y = (Phi_next[i, j + 1] - Phi_next[i, j - 1]) / (2 * dY)

                # Update Phi
                Phi_current[i, j] = Phi_next[i, j] + dt * (
                    -alpha * Phi_W
                    + a_WW * Phi_WW
                    + a_YY * Phi_YY
                    + a_WY * Phi_WY
                    - b_W[i] * Phi_W
                    - b_Y[j] * Phi_Y
                )

        # Apply boundary conditions
        Phi_current[0, :] = Phi_current[1, :]
        Phi_current[-1, :] = Phi_current[-2, :]
        Phi_current[:, 0] = Phi_current[:, 1]
        Phi_current[:, -1] = Phi_current[:, -2]

        # Update Phi
        Phi[:, :, n] = Phi_current

    # Compute V with option
    V = -np.log(-Phi) / gamma
    V_with_option = V[i_W0, j_Y0, 0]

    # Repeat without option
    for i in range(N_W):
        for j in range(N_Y):
            Phi[i, j, -1] = -np.exp(-gamma * W_grid[i])  # No option payoff

    for n in reversed(range(N_t - 1)):
        Phi_next = Phi[:, :, n + 1].copy()
        Phi_current = Phi[:, :, n].copy()

        for i in range(1, N_W - 1):
            for j in range(1, N_Y - 1):
                # Same calculations
                Phi_WW = (Phi_next[i + 1, j] - 2 * Phi_next[i, j] + Phi_next[i - 1, j]) / (dW**2)
                Phi_YY = (Phi_next[i, j + 1] - 2 * Phi_next[i, j] + Phi_next[i, j - 1]) / (dY**2)
                Phi_WY = (Phi_next[i + 1, j + 1] - Phi_next[i + 1, j - 1] - Phi_next[i - 1, j + 1] + Phi_next[i - 1, j - 1]) / (4 * dW * dY)
                Phi_W = (Phi_next[i + 1, j] - Phi_next[i - 1, j]) / (2 * dW)
                Phi_Y = (Phi_next[i, j + 1] - Phi_next[i, j - 1]) / (2 * dY)

                # Update Phi
                Phi_current[i, j] = Phi_next[i, j] + dt * (
                    -alpha * Phi_W
                    + a_WW * Phi_WW
                    + a_YY * Phi_YY
                    + a_WY * Phi_WY
                    - b_W[i] * Phi_W
                    - b_Y[j] * Phi_Y
                )

        # Apply boundary conditions
        Phi_current[0, :] = Phi_current[1, :]
        Phi_current[-1, :] = Phi_current[-2, :]
        Phi_current[:, 0] = Phi_current[:, 1]
        Phi_current[:, -1] = Phi_current[:, -2]

        # Update Phi
        Phi[:, :, n] = Phi_current

    # Compute V without option
    V_no_option = -np.log(-Phi) / gamma
    V_without_option = V_no_option[i_W0, j_Y0, 0]

    # Compute indifference price for this gamma
    indifference_price_gamma = V_without_option - V_with_option
    indifference_prices_gamma.append(indifference_price_gamma)

# Plot indifference price vs risk aversion gamma
plt.figure(figsize=(8, 6))
plt.plot(gamma_values, indifference_prices_gamma, 'o-')
plt.xlabel('Risk Aversion $\\gamma$')
plt.ylabel('Indifference Price $\\pi$')
plt.title('Indifference Price vs Risk Aversion')
plt.grid(True)
plt.show()

# Display results
for g, pi in zip(gamma_values, indifference_prices_gamma):
    print(f"Gamma: {g}, Indifference Price: {pi:.4f}")
