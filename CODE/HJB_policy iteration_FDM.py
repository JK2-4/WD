import numpy as np
from   scipy.sparse import spdiags, identity
from   scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
#from   bokeh.plotting import figure, output_notebook, show, gridplot, save
import pandas as pd
import scipy.stats as ss
import warnings
warnings.filterwarnings("ignore")

CENTRAL =0
FORWARD =1
BACKWARD= 2
np.random.seed(seed=88)

def main():
    r     = 0.03
    sigma = 0.15
    xi    = 0.33
    pi    = 0.1
    W0    = 1.0
    T     = 20.0
    gamma = 14.47
    Wmin  = 0.0
    Wmax  = 5.0
    M     = 1600
    N     = 100
    tol   = 1e-6
    scale = 1.0
    Pmax  = 1.5
    J     = 8

    hsigsq  = 0.5 * sigma ** 2 # half sigma squared -> 0.5 x sigma^2
    sigmaxi = sigma * xi
    dW      = ( Wmax - Wmin) / N
    dt      = T / M
    dWsq    = dW ** 2

    W    = np.linspace( Wmin, Wmax, N + 1 ) # need N+1 for there to be N steps between 0.0 and 5.0
    Ps   = np.linspace( 0.0, Pmax, J ) # discretize controls
    I    = identity( N + 1 )
    Gn   = np.zeros_like( W )
    Gnp1 = np.zeros_like( W )

    W_no    = np.linspace( Wmin, Wmax, N + 1 ) # need N+1 for there to be N steps between 0.0 and 5.0
    Ps_no   = np.linspace( 0.0, Pmax, J ) # discretize controls
    Gn_no   = np.zeros_like( W_no )
    Gnp1_no = np.zeros_like( W_no )


    # Non-traded asset parameters (Y_t)-------------------------------------------------------------------------------------------------------------------------------
    kappa = 0.2
    theta = 42
    sigma = 0.4
    std_asy = np.sqrt(sigma**2 / (2 * kappa))
    paths = 5000  # paths?
    std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))

    #dS/S
    a = 0.4
    b = 0.58
    mu_S = 0.2*a + b    # exposure to Y - drift
    sigma_S = 0.2*sigma
    std_dt_S = 0.2*std_dt
    # ------------------------------------------------------------------------------------------------------------------------------------------------

    BM = ss.norm.rvs(loc=0, scale=1, size=(N-1, 1))
    for t in np.linspace( 0.0, T, 5000):
        tau = T - t

        Y_0 = theta
        Y = Y_0   # OU process backwards in time - Euler Mayurama scheme
        for _ in range(int(tau)):
            Y += kappa + np.exp(-kappa * dt) * (Y - theta) + std_dt * np.float64(BM[int(tau)])

    g = max(0, Y) * np.sqrt(T) # need to discretize as grid as well

    terminal_values = (W - 0.5*gamma)**2 # np.exp (-W * gamma )
    terminal_values_no = (W_no - 0.5*gamma)**2 + g  # np.exp (-W * gamma )


    def bc( t ): # boundary condition
        tau = T - t
        c   = ( 2 * pi ) / r
        BM = ss.norm.rvs(loc=0, scale=1, size=(N-1, 1))

        Y = Y_0   # OU process backwards in time - Euler Mayurama scheme
        for _ in range(int(tau)):
            Y += kappa + np.exp(-kappa * dt) * (Y - theta) + std_dt * np.float64(BM[int(tau)])

        g = max(0, Y) * np.sqrt(tau/T) # need to discretize as grid as well
        e1 = np.exp( r * tau )
        e2 = np.exp( 2 * r * tau )
        alpha = e2 * ( Wmax**2 )
        beta  = ( c * e2 - ( gamma + c ) *e1 ) * Wmax
        delta = ( ( gamma**2 ) / 4.0 ) +  ( ( pi * c ) / ( 2 * r ) ) * ( e2 - 1 )\
                - ( ( pi * ( gamma + c ) ) / r ) * ( e1 - 1 )
        return alpha + beta + delta + g

    def bc_no( t ): # boundary condition
        tau = T - t
        c   = ( 2 * pi ) / r

        e1 = np.exp( r * tau )
        e2 = np.exp( 2 * r * tau )
        alpha = e2 * ( Wmax**2 )
        beta  = ( c * e2 - ( gamma + c ) *e1 ) * Wmax
        delta = ( ( gamma**2 ) / 4.0 ) +  ( ( pi * c ) / ( 2 * r ) ) * ( e2 - 1 ) - ( ( pi * ( gamma + c ) ) / r ) * ( e1 - 1 )
        return alpha + beta + delta

    def alpha( W, p, dirn = CENTRAL ):
        t1 = hsigsq * (p**2) * (W**2) / dWsq
        t2 = ( pi + W * ( r + p * sigmaxi ) )
        if dirn == CENTRAL:
            return t1 - t2 / ( 2 * dW )
        elif dirn == BACKWARD:
            return t1 - t2 / dW
        elif dirn == FORWARD:
            return t1

    def beta( W, p, dirn = CENTRAL ):
        t1 = hsigsq * (p**2) * (W**2) / dWsq
        t2 = ( pi + W * ( r + p * sigmaxi ) )
        if dirn == CENTRAL:
            return t1 + t2 / (2 *dW)
        elif dirn == FORWARD:
            return t1 + t2 / dW
        elif dirn == BACKWARD:
            return t1

    def makeDiagMat( alphas, betas ):
        d0, dl, d2 = -( alphas + betas ), np.roll( alphas, -1 ), np.roll( betas, 1 )
        d0[-1] = 0.
        dl [-2:] = 0.
        data = np.array( [ d0, dl, d2 ] )
        diags = np.array( [ 0, -1, 1 ] )
        return spdiags( data, diags, N + 1, N + 1 )

    def find_optima1_ctrls( Vhat, t ):

        Fmin = np.tile( np.inf, Vhat.size )

        optdiffs = np.zeros_like( Vhat, dtype = int )
        optP    = np.zeros_like( Vhat )

        alphas  = np.zeros_like( Vhat ) # the final
        betas   = np.zeros_like( Vhat ) # the final
        curDiffs = np.zeros_like( Vhat, dtype = int )

        for p in Ps: # Hnd the optimal control
            alphas[:] = -np.inf
            betas[:] = -np.inf
            curDiffs[:] = CENTRAL

            for diff in [ CENTRAL, FORWARD, BACKWARD ]:
                a = alpha( W, p, diff)
                b = beta( W, p, diff )
                positive_coeff_indices = np.logical_and( a >= 0.0, b >= 0.0 ) == True
                positive_coeff_indices = np.logical_and( positive_coeff_indices, alphas==-np.inf )
                indices = np.where( positive_coeff_indices )

                alphas[ indices ] = a[ indices ]
                betas[ indices ] = b[ indices ]
                curDiffs[ indices ] = diff

            M = makeDiagMat( alphas, betas )
            F = M.dot( Vhat )
            indices = np.where( F < Fmin )

            Fmin[indices] = F[indices ]
            optP[indices] = p
            optdiffs[indices] = curDiffs[ indices ]
        return optP, optdiffs

    def find_optima1_ctrls_no( Vhat_no, t ):

        Fmin = np.tile( np.inf, Vhat_no.size )

        optdiffs = np.zeros_like( Vhat_no, dtype = int )
        optP    = np.zeros_like( Vhat_no )

        alphas  = np.zeros_like( Vhat_no ) # the final
        betas   = np.zeros_like( Vhat_no ) # the final
        curDiffs = np.zeros_like( Vhat_no, dtype = int )

        for p in Ps_no: # Hnd the optimal control
            alphas[:] = -np.inf
            betas[:] = -np.inf
            curDiffs[:] = CENTRAL

            for diff in [ CENTRAL, FORWARD, BACKWARD ]:
                a = alpha( W_no, p, diff)
                b = beta( W_no, p, diff )
                positive_coeff_indices = np.logical_and( a >= 0.0, b >= 0.0 ) == True
                positive_coeff_indices = np.logical_and( positive_coeff_indices, alphas==-np.inf )
                indices = np.where( positive_coeff_indices )

                alphas[ indices ] = a[ indices ]
                betas[ indices ] = b[ indices ]
                curDiffs[ indices ] = diff

            M = makeDiagMat( alphas, betas )
            F = M.dot( Vhat )
            indices = np.where( F < Fmin )

            Fmin[indices] = F[indices ]
            optP[indices] = p
            optdiffs[indices] = curDiffs[ indices ]
        return optP, optdiffs

    timesteps = np.linspace( 0.0, T, M + 1 )[:-1] # drop last item which is T=20.0
    timesteps = np.flipud( timesteps )

    V = terminal_values
    V_no = terminal_values_no

    alphas = np.zeros_like( V )
    betas = np.zeros_like( V )

    alphas_no = np.zeros_like( V_no )
    betas_no  = np.zeros_like( V_no )

    for t in timesteps:

        Vhat = V.copy()
        Gnp1[-1] =bc(t+ dt)
        Gn[-1] = bc( t)
        B        = Gn - Gnp1 # new boundary cone - old boundary cone


        while True:
            ctrls, diffs = find_optima1_ctrls( Vhat, t )
            for diff in [ CENTRAL, FORWARD, BACKWARD ]:
                indices = np.where( diffs == diff )
                alphas[indices] = alpha( W[indices], ctrls[indices], diff )
                betas[indices] = beta( W[indices], ctrls[indices], diff )

            A   = makeDiagMat( alphas, betas )
            M = I - dt * A
            Vnew = spsolve( M, V + B )
            scale    = np.maximum( np.abs( Vnew ), np.ones_like( Vnew ) )
            residuals = np.abs( Vnew - Vhat ) / scale
            if np.all( residuals[:-1] < tol ):
                V = Vnew
                break
            else:
                Vhat = Vnew

        Vhat_no = V_no.copy()
        Gnp1_no[-1] =bc_no(t+ dt)
        Gn_no[-1] = bc_no( t)
        B_no        = Gn_no - Gnp1_no # new boundary cone - old boundary cone

        while True:
            ctrls_no, diffs_no = find_optima1_ctrls_no( Vhat_no, t )
            for diff_no in [ CENTRAL, FORWARD, BACKWARD ]:
                indices_no = np.where( diffs_no == diff_no )
                alphas_no[indices_no] = alpha( W_no[indices_no], ctrls_no[indices_no], diff_no )
                betas_no[indices_no] = beta( W_no[indices_no], ctrls_no[indices_no], diff_no )

            A_no   = makeDiagMat( alphas_no, betas_no )
            M_no = I - dt * A_no
            Vnew_no = spsolve( M_no, V_no + B_no )
            scale_no    = np.maximum( np.abs( Vnew_no ), np.ones_like( Vnew_no ) )
            residuals_no = np.abs( Vnew_no - Vhat_no ) / scale_no
            if np.all( residuals_no[:-1] < tol ):
                V_no = Vnew_no
                break
            else:
                Vhat_no = Vnew_no

    return W, V, ctrls, W_no, V_no, ctrls_no

W, V, ctrls, W_no, V_no, ctrls_no = main()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
ax1.plot(W, V, color = 'pink', lw=2)
ax1.set_title('Plot of Value$ with no derivative against wealth')
ax2.plot(W, ctrls, color = 'orange', lw=2, alpha = 0.7)
_ = ax2.set_title('Plot of optimal control against wealth')


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
ax1.plot(W_no, V_no, color = 'pink', lw=2)
ax1.set_title('Plot of Value with derivative against wealth')
ax2.plot(W_no, ctrls_no, color = 'orange', lw=2, alpha = 0.7)
_ = ax2.set_title('Plot of optimal control against wealth')


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
ax1.plot(W, V_no/V, color = 'pink', lw=2)
ax1.set_title('Plot of $V_no(W_no=w_0, 0)$ against wealth')
ax2.plot(W, ctrls_no/ctrls, color = 'orange', lw=2, alpha = 0.7)
_ = ax2.set_title('Plot of optimal control against wealth')

########################## TO DEBUG

factor = 1/gamma
indiff = factor*(np.log(V_no/V))
indiff = np.where(indiff < 0, 0, indiff)  # Avoid division by zero

f, (ax1) = plt.subplots(1, figsize=(16,5))
ax1.plot(V_no, indiff, color = 'pink', lw=2)
ax1.set_title('Indifference Price against wealth')
# f.savefig(path1,  bbox_inches='tight')

from sympy import *
init_printing()

print('Indifference Price at t=0 is' ,indiff[0])
