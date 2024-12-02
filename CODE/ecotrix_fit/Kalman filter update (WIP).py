import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mpl_toolkits import mplot3d
from matplotlib import cm
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess
import pandas as pd



class OU_process:
    """
    Class for the OU process:
    theta = long term mean
    sigma = diffusion coefficient
    kappa = mean reversion coefficient
    """

    def __init__(self, sigma=0.2, theta=-0.1, kappa=0.1):
        self.theta = theta
        if sigma < 0 or kappa < 0:
            raise ValueError("sigma,theta,kappa must be positive")
        else:
            self.sigma = sigma
            self.kappa = kappa

    def path(self, X0=0, T=1, N=10000, paths=1):
        """
        Produces a matrix of OU process:  X[N, paths]
        X0 = starting point
        N = number of time points (there are N-1 time steps)
        T = Time in years
        paths = number of paths
        """

        dt = T / (N - 1)
        X = np.zeros((N, paths))
        X[0, :] = X0
        W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

        std_dt = np.sqrt(self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * dt)))
        for t in range(0, N - 1):
            X[t + 1, :] = self.theta + np.exp(-self.kappa * dt) * (X[t, :] - self.theta) + std_dt * W[t, :]

        return X


###############3 Simulation with fitted paramms 

np.random.seed(seed=42)

N = 20000  # time steps
paths = 5000  # number of paths
T = 5
T_vec, dt = np.linspace(0, T, N, retstep=True)

kappa = 3
theta = 0.5
sigma = 0.5
std_asy = np.sqrt(sigma**2 / (2 * kappa))  # asymptotic standard deviation

X0 = 2
X = np.zeros((N, paths))
X[0, :] = X0
W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

# Uncomment for Euler Maruyama
# for t in range(0,N-1):
#    X[t + 1, :] = X[t, :] + kappa*(theta - X[t, :])*dt + sigma * np.sqrt(dt) * W[t, :]

std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
for t in range(0, N - 1):
    X[t + 1, :] = theta + np.exp(-kappa * dt) * (X[t, :] - theta) + std_dt * W[t, :]

X_T = X[-1, :]  # values of X at time T
X_1 = X[:, 1]  # a single path

 

mean_T = theta + np.exp(-kappa * T) * (X0 - theta)
std_T = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * T)))

param = ss.norm.fit(X_T)  # FIT from data
print(f"Theoretical mean={mean_T.round(6)} and theoretical STD={std_T.round(6)}")
print("Parameters from the fit: mean={0:.6f}, STD={1:.6f}".format(*param))  # these are MLE parameters

######### plot 

N_processes = 10  # number of processes
x = np.linspace(X_T.min(), X_T.max(), 100)
pdf_fitted = ss.norm.pdf(x, *param)

fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(T_vec, X[:, :N_processes], linewidth=0.5)
ax1.plot(T_vec, (theta + std_asy) * np.ones_like(T_vec), label="1 asymptotic std dev", color="black")
ax1.plot(T_vec, (theta - std_asy) * np.ones_like(T_vec), color="black")
ax1.plot(T_vec, theta * np.ones_like(T_vec), label="Long term mean")
ax1.legend(loc="upper right")
ax1.set_title(f"{N_processes} OU processes")
ax1.set_xlabel("T")
ax2.plot(x, pdf_fitted, color="r", label="Normal density")
ax2.hist(X_T, density=True, bins=50, facecolor="LightBlue", label="Frequency of X(T)")
ax2.legend()
ax2.set_title("Histogram vs Normal distribution")
ax2.set_xlabel("X(T)")
plt.show()


n1 = 5950
n2 = 6000
t1 = n1 * dt
t2 = n2 * dt

cov_th = sigma**2 / (2 * kappa) * (np.exp(-kappa * np.abs(t1 - t2)) - np.exp(-kappa * (t1 + t2)))
print(f"Theoretical COV[X(t1), X(t2)] = {cov_th.round(4)} with t1 = {t1.round(4)} and t2 = {t2.round(4)}")
print(f"Computed covariance from data COV[X(t1), X(t2)] = {np.cov( X[n1, :], X[n2, :] )[0,1].round(4)}")

########################### OLS fit 
XX = X_1[:-1]
YY = X_1[1:]
beta, alpha, _, _, _ = ss.linregress(XX, YY)  # OLS
kappa_ols = -np.log(beta) / dt
theta_ols = alpha / (1 - beta)
res = YY - beta * XX - alpha  # residuals
std_resid = np.std(res, ddof=2)
sig_ols = std_resid * np.sqrt(2 * kappa_ols / (1 - beta**2))

print("OLS theta = ", theta_ols)
print("OLS kappa = ", kappa_ols)
print("OLS sigma = ", sig_ols)


############################### MLE
Sx = np.sum(XX)
Sy = np.sum(YY)
Sxx = XX @ XX
Sxy = XX @ YY
Syy = YY @ YY

theta_mle = (Sy * Sxx - Sx * Sxy) / (N * (Sxx - Sxy) - (Sx**2 - Sx * Sy))
kappa_mle = -(1 / dt) * np.log(
    (Sxy - theta_mle * Sx - theta_mle * Sy + N * theta_mle**2) / (Sxx - 2 * theta_mle * Sx + N * theta_mle**2)
)
sigma2_hat = (
    Syy
    - 2 * np.exp(-kappa_mle * dt) * Sxy
    + np.exp(-2 * kappa_mle * dt) * Sxx
    - 2 * theta_mle * (1 - np.exp(-kappa_mle * dt)) * (Sy - np.exp(-kappa_mle * dt) * Sx)
    + N * theta_mle**2 * (1 - np.exp(-kappa_mle * dt)) ** 2
) / N
sigma_mle = np.sqrt(sigma2_hat * 2 * kappa_mle / (1 - np.exp(-2 * kappa_mle * dt)))

print("theta MLE = ", theta_mle)
print("kappa MLE = ", kappa_mle)
print("sigma MLE = ", sigma_mle)

############################### hitting time and new density

T_to_theta = np.argmax(X <= theta if (X0 > theta) else X >= theta, axis=0) * dt  # first passage time
print(f"The expected time from X0 to theta is: {T_to_theta.mean()} with std error: {ss.sem(T_to_theta)}")
print("The standard deviation of the first time the process touches theta is: ", T_to_theta.std())

def density_T_to_theta(t, C):
    return (
        np.sqrt(2 / np.pi)
        * np.abs(C)
        * np.exp(-t)
        / (1 - np.exp(-2 * t)) ** (3 / 2)
        * np.exp(-((C**2) * np.exp(-2 * t)) / (2 * (1 - np.exp(-2 * t))))
    )
C = (X0 - theta) * np.sqrt(2 * kappa) / sigma  # new starting point
fig = plt.figure(figsize=(10, 4))
x = np.linspace(T_to_theta.min(), T_to_theta.max(), 100)
plt.plot(x, kappa * density_T_to_theta(kappa * x, C), color="red", label="OU hitting time density")
plt.hist(T_to_theta, density=True, bins=100, facecolor="LightBlue", label="frequencies of T")
plt.title("First passage time distribution from X0 to theta")
plt.legend()
plt.show()
theoretical_T = quad(lambda t: t * kappa * density_T_to_theta(kappa * t, C), 0, 1000)[0]
theoretical_std = np.sqrt(
    quad(lambda t: (t - theoretical_T) ** 2 * kappa * density_T_to_theta(kappa * t, C), 0, 1000)[0]
)
print("Theoretical expected hitting time: ", theoretical_T)
print("Theoretical standard deviation of the hitting time: ", theoretical_std)

############################# Bond price analytical

B = 1 / kappa * (1 - np.exp(-kappa * T))
A = np.exp((theta - sigma**2 / (2 * kappa**2)) * (B - T) - sigma**2 / (4 * kappa) * B**2)
P = A * np.exp(-B * X0)
print("Vasicek bond price: ", P)

############################# Bond price MC
disc_factor = np.exp(-X.mean(axis=0) * T)
P_MC = np.mean(disc_factor)
st_err = ss.sem(disc_factor)
print(f"Vasicek bond price by MC: {P_MC} with std error: {st_err}")


############################# Bond price PDE (FDM - Upwind scheme)

Nspace = 6000  # M space steps
Ntime = 6000  # N time steps
r_max = 3  # A2
r_min = -0.8  # A1
r, dr = np.linspace(r_min, r_max, Nspace, retstep=True)  # space discretization
T_array, Dt = np.linspace(0, T, Ntime, retstep=True)  # time discretization
Payoff = 1  # Bond payoff

V = np.zeros((Nspace, Ntime))  # grid initialization
offset = np.zeros(Nspace - 2)  # vector to be used for the boundary terms
V[:, -1] = Payoff  # terminal conditions
V[-1, :] = np.exp(-r[-1] * (T - T_array))  # lateral boundary condition
V[0, :] = np.exp(-r[0] * (T - T_array))  # lateral boundary condition

# construction of the tri-diagonal matrix D
sig2 = sigma * sigma
drr = dr * dr
max_part = np.maximum(kappa * (theta - r[1:-1]), 0)  # upwind positive part
min_part = np.minimum(kappa * (theta - r[1:-1]), 0)  # upwind negative part

a = min_part * (Dt / dr) - 0.5 * (Dt / drr) * sig2
b = 1 + Dt * r[1:-1] + (Dt / drr) * sig2 + Dt / dr * (max_part - min_part)
c = -max_part * (Dt / dr) - 0.5 * (Dt / drr) * sig2

a0 = a[0]
cM = c[-1]  # boundary terms
aa = a[1:]
cc = c[:-1]  # upper and lower diagonals
D = sparse.diags([aa, b, cc], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()  # matrix D

for n in range(Ntime - 2, -1, -1):
    # backward computation
    offset[0] = a0 * V[0, n]
    offset[-1] = cM * V[-1, n]
    V[1:-1, n] = spsolve(D, (V[1:-1, n + 1] - offset))

# finds the bond price with initial value X0
Price = np.interp(X0, r, V[:, 0])
print("The Vasicek Bond price by PDE is: ", Price)



##################################### plot
fig = plt.figure(figsize=(15, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection="3d")

ax1.text(X0, 0.06, "Bond Price")
ax1.plot([X0, X0], [0, Price], "k--")
ax1.plot([-0.5, X0], [Price, Price], "k--")
ax1.plot(r, V[:, 0], color="red", label="Barrier curve")
ax1.set_xlim(-0.4, 2.5)
ax1.set_ylim(0.025, 0.12)
ax1.set_xlabel("r")
ax1.set_ylabel("P")
ax1.legend(loc="upper right")
ax1.set_title("Vasicek bond price at t=0")

X_plt, Y_plt = np.meshgrid(T_array, r[700:-200])  # I consider [700:-200] to remove lateral boundary effects
ax2.plot_surface(Y_plt, X_plt, V[700:-200])
ax2.set_title("Vasicek bond price surface")
ax2.set_xlabel("r")
ax2.set_ylabel("time")
ax2.set_zlabel("V")
plt.show()



#####################################  Kalman filter - state space vs observability 

# add noise to simulation     # state = alpha + beta(x[-1]) + true residual var ; obs = true + guassian noise
sig_eta = std_resid
var_eta = sig_eta**2  # error of the true state process
sig_eps = 0.1
var_eps = sig_eps**2  # error of the measurement
np.random.seed(seed=42)
eps = ss.norm.rvs(loc=0, scale=sig_eps, size=N)  # additional gaussian noise
eps[0] = 0
Y_1 = X_1 + eps  # process + noise = measurement process

#observ vs true DGP - simple case (constant matrix coeff case)

fig = plt.figure(figsize=(13, 4))
plt.plot(T_vec, X_1, linewidth=0.5, alpha=1, label="True process", color="#1f77b4")
plt.plot(T_vec, theta * np.ones_like(T_vec), label="Long term mean", color="#d62728")
plt.plot(T_vec, Y_1, linewidth=0.3, alpha=0.5, label="Measurement process = true + noise", color="#1f77b4")
plt.legend()
plt.title("OU process plus some noise")
plt.xlabel("time")
plt.show()

# step 1 : kalman fn define

def Kalman(Y, x0, P0, alpha, beta, var_eta, var_eps):
    """Kalman filter algorithm for the OU process."""

    N = len(Y)  # length of measurements
    xs = np.zeros_like(Y)  # Initialization
    Ps = np.zeros_like(Y)

    x = x0
    P = P0  # initial values of h and P
    log_2pi = np.log(2 * np.pi)
    loglikelihood = 0  # initialize log-likelihood

    for k in range(N):
        # Prediction step
        x_p = alpha + beta * x  # predicted h
        P_p = beta**2 * P + var_eta  # predicted P

        # auxiliary variables
        r = Y[k] - x_p  # residual
        S = P_p + var_eps
        KG = P_p / S  # Kalman Gain

        # Update step
        x = x_p + KG * r
        P = P_p * (1 - KG)

        loglikelihood += -0.5 * (log_2pi + np.log(S) + (r**2 / S))
        xs[k] = x
        Ps[k] = P

    return xs, Ps, loglikelihood

# step 2 - MLE

skip_data = 1000
training_size = 5000
train = Y_1[skip_data : skip_data + training_size]
test = Y_1[skip_data + training_size :]

## Initial guess for parameters
guess_beta, guess_alpha, _, _, _ = ss.linregress(train[1:], train[:-1])
guess_var_eps = np.var(train[:-1] - guess_beta * train[1:] - guess_alpha, ddof=2)
def minus_likelihood(c):
    """Returns the negative log-likelihood"""
    _, _, loglik = Kalman(train, X0, 10, c[0], c[1], c[2], c[3])
    return -loglik


result = minimize(
    minus_likelihood,
    x0=[guess_alpha, guess_beta, 0.01, guess_var_eps],
    method="L-BFGS-B",
    bounds=[[-1, 1], [1e-15, 1], [1e-15, 1], [1e-15, 1]],
    tol=1e-12,
)
kalman_params = result.x
alpha_KF = kalman_params[0]
beta_KF = kalman_params[1]
var_eta_KF = kalman_params[2]
var_eps_KF = kalman_params[3]
print(result)
print(f"Error in estimation of alpha = {(np.abs(alpha-alpha_KF)/alpha *100).round(2)}%")
print(f"Error in estimation of beta = {(np.abs(beta-beta_KF)/beta *100).round(4)}%")
print(f"Error in estimation of var eta = {(np.abs(var_eta-var_eta_KF)/var_eta *100).round(2)}%")
print(f"Error in estimation of var eps = {(np.abs(var_eps-var_eps_KF)/var_eps *100).round(2)}%")

# fixed interval RTS smoother 
def Smoother(Y, x0, P0, alpha, beta, var_eta, var_eps):
    """Kalman smoother"""

    xs, Ps, _ = Kalman(Y, x0, P0, alpha, beta, var_eta, var_eps)

    xs_smooth = np.zeros_like(xs)
    Ps_smooth = np.zeros_like(Ps)
    Cs_smooth = np.zeros_like(Ps)
    C = np.zeros_like(Ps)
    xs_smooth[-1] = xs[-1]
    Ps_smooth[-1] = Ps[-1]
    K = (beta**2 * Ps[-2] + var_eta) / (beta**2 * Ps[-2] + var_eta + var_eps)
    Cs_smooth[-1] = Ps[-1]
    Cs_smooth[-2] = beta * (1 - K) * Ps[-2]

    for k in range(len(xs) - 2, -1, -1):
        C[k] = beta * Ps[k] / (beta**2 * Ps[k] + var_eta)
        xs_smooth[k] = xs[k] + C[k] * (xs_smooth[k + 1] - (alpha + xs[k] * beta))
        Ps_smooth[k] = Ps[k] + C[k] ** 2 * (Ps_smooth[k + 1] - (beta**2 * Ps[k] + var_eta))
        if k == (len(xs) - 2):
            continue
        Cs_smooth[k] = C[k] * Ps[k + 1] + C[k] * C[k + 1] * (
            Cs_smooth[k + 1] - beta * Ps[k + 1]
        )  # covariance x(k) and x(k+1)
    return xs_smooth, Ps_smooth, Cs_smooth
x_tmp, P_tmp, _ = Kalman(train, 1, 10, alpha_KF, beta_KF, var_eta_KF, var_eps_KF)  # to get initial values for KF
xs, Ps, _ = Kalman(test, x_tmp[-1], P_tmp[-1], alpha_KF, beta_KF, var_eta_KF, var_eps_KF)
x_smooth, P_smooth, _ = Smoother(test, x_tmp[-1], P_tmp[-1], alpha_KF, beta_KF, var_eta_KF, var_eps_KF)
V_up = xs + np.sqrt(Ps)  # error up bound
V_down = xs - np.sqrt(Ps)  # error down bound


################################## Kalman results 
fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(
    T_vec[skip_data + training_size :], xs, linewidth=0.5, alpha=1, label="Optimal state estimate", color="#1f77b4"
)
ax1.plot(T_vec[skip_data + training_size :], theta * np.ones_like(xs), label="Long term mean", color="#d62728")
ax1.plot(
    T_vec[skip_data + training_size :],
    Y_1[skip_data + training_size :],
    linewidth=0.6,
    alpha=0.5,
    label="Process + noise",
    color="#1f77b4",
)
ax1.plot(
    T_vec[skip_data + training_size :],
    x_smooth,
    linewidth=0.5,
    alpha=1,
    label="Smoothed state estimate",
    color="purple",
)
ax1.fill_between(
    x=T_vec[skip_data + training_size :],
    y1=V_up,
    y2=V_down,
    alpha=0.7,
    linewidth=2,
    color="seagreen",
    label="Kalman error: $\pm$ 1 std dev ",
)
ax1.legend()
ax1.set_title("State estimation of the OU process")
ax1.set_xlabel("time")
ax2.plot(T_vec[skip_data + training_size :], xs, linewidth=1, alpha=1, label="Optimal state estimator", color="blue")
ax2.plot(T_vec[skip_data + training_size :], theta * np.ones_like(xs), label="Long term mean", color="#d62728")
ax2.plot(
    T_vec[skip_data + training_size :],
    Y_1[skip_data + training_size :],
    linewidth=0.5,
    alpha=0.8,
    label="Process + noise",
    color="#1f77b4",
)
ax2.plot(
    T_vec[skip_data + training_size :], x_smooth, linewidth=1, alpha=1, label="Smoothed state estimate", color="purple"
)
ax2.fill_between(
    x=T_vec[skip_data + training_size :],
    y1=V_up,
    y2=V_down,
    alpha=0.7,
    linewidth=2,
    color="seagreen",
    label="Kalman error: $\pm$ 1 std dev ",
)
ax2.set_xlim(1.99, 2.15)
ax2.set_ylim(0.2, 0.65)
ax2.legend()
ax2.set_title("Zoom")
ax2.set_xlabel("time")
plt.show()

print("Test set, mean linear error Kalman: ", np.linalg.norm(xs - X_1[skip_data + training_size :], 1) / len(xs))
print("Average standard deviation of the estimate: ", np.mean(np.sqrt(Ps)))
print("Test set, MSE Kalman: ", np.linalg.norm(xs - X_1[skip_data + training_size :], 2) ** 2 / len(xs))
print("Test set, MSE Smoother: ", np.linalg.norm(x_smooth - X_1[skip_data + training_size :], 2) ** 2 / len(x_smooth))


