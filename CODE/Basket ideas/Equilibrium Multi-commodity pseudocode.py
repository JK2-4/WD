import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import norm

np.random.seed(42)
N = 10000   
mu_T = 80  # (daily average temperature)  
sigma_T = 5      
strike = 85       
B1 = 1.0 # one period with risk free rate r=0
PR = 120.0 # Retail price
nu_buyer = 0.1   # Risk aversion 
nu_issuer = 0.01 

# -----------------------------
# dummy params and monte carlo 
# -----------------------------
Z = np.random.randn(N)
T = mu_T + sigma_T * Z
W1 = np.maximum(T - strike, 0) #WD

# W1 sample moments (for supply function)
mu_W1 = np.mean(W1)
var_W1 = np.var(W1)
print(f"Simulated Weather Option Payoff: mean = {mu_W1:.2f}, variance = {var_W1:.2f}")

# Each buyer_i - lognormal models:
#   D_i = exp( a_i * Z + b_i * Z_i + c_i )
#   P_i = exp( e_i * Z + f_i * Z_i + g_i )
# Z is the common shock (correlated with temperature) and Z_i is an idiosyncratic shock for buyer i.

# correlation among T, D and P : no commodity derivative assumption
params = {
    1: {'a': 0.3, 'b': 0.2, 'c': 16.0, 'e': 0.2, 'f': 0.1, 'g': 4.6}, # Buyer 1: positive correl
    # Buyer 2: negative
    2: {'a': -0.3, 'b': 0.2, 'c': 16.0, 'e': -0.2, 'f': 0.1, 'g': 4.6},
    # Buyer 3: positive 
    3: {'a': 0.4, 'b': 0.3, 'c': 16.0, 'e': 0.3, 'f': 0.2, 'g': 4.6},
    # Buyer 4: negative
    4: {'a': -0.4, 'b': 0.3, 'c': 16.0, 'e': -0.3, 'f': 0.2, 'g': 4.6}
}
D = {}   # commodity demand
P = {}   # spot price
I = {}   # income function: I = (PR - P) * D

for i in range(1, 5):
    par = params[i]
    # Independent idiosyncratic shock for buyer i
    Zi = np.random.randn(N)
    # Demand and spot price simulation
    D[i] = np.exp(par['a'] * Z + par['b'] * Zi + par['c'])
    P[i] = np.exp(par['e'] * Z + par['f'] * Zi + par['g'])
    I[i] = (PR - P[i]) * D[i]  # income: (retail price - spot price)*demand

    # Print approximate means for reference:
    print(f"Buyer {i}: Mean Demand = {np.mean(D[i]):.2e}, Mean Spot Price = {np.mean(P[i]):.2f}")

# ----------------------------- linear Demand and Supply Functions for Weather Derivatives -----------------------------

# buyer_i's weather derivative demand as linear functions:
#   alpha_i(W0) = A_i - B_i * W0.
# choose A_i and B_i such that:
#   - Buyers 1 and 4 take short positions (negative A_i)
#   - Buyers 2 and 3 take long positions.
A = {1: -1.0, 2:  1.5, 3: 2.5, 4: -2.0}
B_coef = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.10}

def buyer_demand(W0, i):
    """Weather derivative demand for buyer i as a linear function."""
    return A[i] - B_coef[i] * W0

# issuer: linear supply function:
#   alpha_m(W0) = C * W0 + D.
C = 0.50
D_const = -1.0
def issuer_supply(W0):
    return C * W0 + D_const

# Market clearing requires: sum_{i=1}^4 alpha_i(W0) = issuer_supply(W0).

def market_clearing(W0):
    total_demand = sum(buyer_demand(W0, i) for i in range(1, 5))
    supply = issuer_supply(W0)
    return total_demand - supply

# price solve
W0_eq, = fsolve(market_clearing, x0=1.0)
print(f"\nEquilibrium weather derivative price (W0) = {W0_eq:.2f}")

# individual optimal choices given equilibrium price:
alpha_opt = {i: buyer_demand(W0_eq, i) for i in range(1, 5)}
alpha_issuer = issuer_supply(W0_eq)
print("Optimal positions (weather derivative quantities):")
for i in range(1, 5):
    print(f"  Buyer {i}: alpha = {alpha_opt[i]:.2f}")
print(f"  Issuer: alpha = {alpha_issuer:.2f}")

# Plot Equilibrium Price vs. Actuarial Price and Optimal Choices
# Define an "actuarial" price as the discounted expected payoff under P, i.e. the mean of W1 (since B1=1. this is the E[W1] discounted.
W0_actuarial = mu_W1

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.axhline(W0_eq, color='red', linestyle='--', label=f'Equilibrium Price = {W0_eq:.2f}')
plt.axhline(W0_actuarial, color='blue', linestyle=':', label=f'Actuarial Price = {W0_actuarial:.2f}')
plt.title("Weather Derivative Price Comparison")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
buyers = np.arange(1,5)
positions = [alpha_opt[i] for i in buyers]
plt.bar(buyers, positions, color=['red','green','blue','orange'])
plt.xlabel("Buyer")
plt.ylabel("Optimal Position (alpha)")
plt.title("Optimal Weather Derivative Positions")
plt.xticks(buyers)
plt.grid(True)

plt.tight_layout()
plt.show()

# Aggregated Demand and Supply Curves
W0_vals = np.linspace(0, 5, 100)
demand_vals = [sum(buyer_demand(w0, i) for i in range(1,5)) for w0 in W0_vals]
supply_vals = [issuer_supply(w0) for w0 in W0_vals]

plt.figure(figsize=(6,5))
plt.plot(W0_vals, demand_vals, label="Aggregate Demand", color='blue')
plt.plot(W0_vals, supply_vals, label="Issuer Supply", color='red')
plt.xlabel("Weather Derivative Price (W0)")
plt.ylabel("Quantity")
plt.title("Aggregate Demand and Supply Curves")
plt.legend()
plt.grid(True)
plt.show()

# Risk Hedging and Risk Sharing Effects - change in the "certain equivalent"
# simulated profit distributions for two representative buyers under three cases:
#   (i) Before hedging (no weather and commodity exposure): Profit = I_i.
#   (ii) Including commodity derivatives (assume hedging reduces variance by 20%).
#   (iii) Including commodity + weather derivatives (assume further variance reduction by 10%).

def adjust_profit(profit, reduction):
    """Assume hedging reduces variance by shifting the profit distribution closer to its mean.
       'reduction' is a fraction by which the deviation from the mean is reduced."""
    mean_profit = np.mean(profit)
    return mean_profit + (profit - mean_profit) * (1 - reduction)

# buyer 1 (has commodity derivatives) and buyer 3 (no commodity derivatives)
buyers_to_plot = [1, 3]
labels = {1: "Buyer 1", 3: "Buyer 3"}

plt.figure(figsize=(12,5))
for j, buyer in enumerate(buyers_to_plot, 1):
    profit_before = I[buyer]  # unhedged
    profit_comm = adjust_profit(profit_before, reduction=0.20)
    profit_comm_wea = adjust_profit(profit_comm, reduction=0.10)
    
    plt.subplot(1,2,j)
    bins = np.linspace(np.percentile(profit_before, 1), np.percentile(profit_before, 99), 50)
    plt.hist(profit_before, bins=bins, density=True, alpha=0.5, label="Before Hedge")
    plt.hist(profit_comm, bins=bins, density=True, alpha=0.5, label="With Commodity Hedge")
    plt.hist(profit_comm_wea, bins=bins, density=True, alpha=0.5, label="With Comm.+Weather Hedge")
    plt.xlabel("Profit")
    plt.ylabel("Density")
    plt.title(f"Profit Distribution for {labels[buyer]}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

#  Optimal Commodity Derivatives Portfolio Payoff
# in paper, the optimal commodity derivatives payoff (x*(P)) is given by - 
#   x*(P) = μ_I - E[I|P] - (α* W0 / B1) * (1 + (1/(W0 B1))*(E[W1|P] - μ_W1))
# approx by first binning the simulated spot prices for Buyer 1, and computing the conditional average of I and W1.

buyer = 1  # for example, Buyer 1
P_vals = P[buyer]
I_vals = I[buyer]
W1_vals = W1  # common for all

# Bin P (spot price) values and compute conditional means.
n_bins = 30
bins = np.linspace(np.percentile(P_vals, 1), np.percentile(P_vals, 99), n_bins)
bin_indices = np.digitize(P_vals, bins)
E_I_given_P = np.array([np.mean(I_vals[bin_indices==i]) if np.sum(bin_indices==i)>0 else np.nan for i in range(1, n_bins+1)])
E_W1_given_P = np.array([np.mean(W1_vals[bin_indices==i]) if np.sum(bin_indices==i)>0 else np.nan for i in range(1, n_bins+1)])
P_bin_centers = (bins[:-1] + bins[1:]) / 2
# For simplicity, let μ_I be the overall mean of I and use our computed μ_W1.
mu_I = np.mean(I_vals)

# Using (with B1=1) – as an approximation.
x_star = (mu_I - E_I_given_P) - (alpha_opt[buyer] * W0_eq) * (1 + (1/(W0_eq))*(E_W1_given_P - mu_W1))

plt.figure(figsize=(6,5))
plt.plot(P_bin_centers, x_star[:-1], marker='o')
plt.xlabel("Spot Price (P)")
plt.ylabel("Optimal Commodity Derivatives Payoff x*(P)")
plt.title("Approximate Optimal Commodity Derivatives Payoff (Buyer 1)")
plt.grid(True)
plt.show()
