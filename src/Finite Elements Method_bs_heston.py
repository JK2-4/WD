import numpy as np
from numpy import sqrt, exp, log, pi, sin, linspace
import scipy.stats as st
from mpi4py import MPI
import ufl

from dolfinx import mesh, fem, plot
from dolfinx.fem import (Function, functionspace, locate_dofs_geometrical, 
                         dirichletbc, Constant, Expression)
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_interval, create_rectangle
from petsc4py import PETSc

# Define standard normal cumulative distribution function
N = st.norm.cdf

# Black-Scholes option pricing formula
def bs(s0, K, sigma, r, T, q=0, cp=1):
    # cp=1: call, cp=-1: put
    d1 = (log(s0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return cp * s0 * exp(-q * T) * N(cp * d1) - cp * K * exp(-r * T) * N(cp * d2)

# Double Barrier Option pricing under Black-Scholes Model
def dbo_bs_anly(sigma, r, T, L, U, S, X, cp=1, q=0, delta1=0, delta2=0, n_terms=10):
    b = r if q == 0 else r - q

    def mu1(n):
        return 2 * (b - delta2 - n * (delta1 - delta2)) / (sigma**2) + 1

    def mu2(n):
        return 2 * n * (delta1 - delta2) / (sigma**2)

    def mu3(n):
        return 2 * (b - delta2 + n * (delta1 - delta2)) / (sigma**2) + 1

    F = U * exp(delta1 * T)
    E = L * exp(delta2 * T)

    def d1(n):
        return (log((S * U**(2*n)) / (X * L**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))

    def d2(n):
        return (log((S * U**(2*n)) / (F * L**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))

    def d3(n):
        return (log((L**(2*n + 2)) / (X * S * U**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))

    def d4(n):
        return (log((L**(2*n + 2)) / (F * S * U**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))

    def y1(n):
        return (log((S * U**(2*n)) / (E * L**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))

    def y2(n):
        return (log((S * U**(2*n)) / (X * L**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))

    def y3(n):
        return (log((L**(2*n + 2)) / (E * S * U**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))

    def y4(n):
        return (log((L**(2*n + 2)) / (X * S * U**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))

    if cp == 1:  # Call option
        term1 = sum([
            ((U**n) / (L**n))**mu1(n) * (L / S)**mu2(n) * (N(d1(n)) - N(d2(n))) -
            ((L**(n+1)) / (U**n * S))**mu3(n) * (N(d3(n)) - N(d4(n)))
            for n in range(-n_terms, n_terms + 1)
        ])
        term2 = sum([
            ((U**n)/(L**n))**(mu1(n) - 2) * (L/S)**mu2(n) * (N(d1(n) - sigma * sqrt(T)) - N(d2(n) - sigma * sqrt(T))) -
            ((L**(n+1)) / (U**n * S))**(mu3(n) - 2) * (N(d3(n) - sigma * sqrt(T)) - N(d4(n) - sigma * sqrt(T)))
            for n in range(-n_terms, n_terms + 1)
        ])
        return (S * exp((b - r) * T) * term1) - (X * exp(-r * T) * term2)
    else:  # Put option
        term1 = sum([
            ((U**n)/(L**n))**(mu1(n) - 2) * (L/S)**mu2(n) * (N(y1(n) - sigma * sqrt(T)) - N(y2(n) - sigma * sqrt(T))) -
            ((L**(n+1)) / (U**n * S))**mu3(n) * (N(y3(n) - sigma * sqrt(T)) - N(y4(n) - sigma * sqrt(T)))
            for n in range(-n_terms, n_terms + 1)
        ])
        term2 = sum([
            ((U**n) / (L**n))**mu1(n) * (L / S)**mu2(n) * (N(y1(n)) - N(y2(n))) -
            ((L**(n+1)) / (U**n * S))**mu3(n) * (N(y3(n)) - N(y4(n)))
            for n in range(-n_terms, n_terms + 1)
        ])
        return (X * exp(-r * T) * term1) - (S * exp((b - r) * T) * term2)

# Finite Element Method (FEM) implementation using FEniCSx for Double Barrier Option Pricing
def dbo_bs_fem(s0, K, sigma, r, T, dt, lb, ub, cp=1):
    # Create a 1D mesh from lb to ub with n_el elements
    n_el = 1000
    mesh_domain = mesh.create_interval(MPI.COMM_WORLD, n_el, np.array([lb, ub]))
    
    # Define function space (quadratic Lagrange elements)
    V = fem.functionspace(mesh_domain, ("Lagrange", 2))
    
    # Define boundary conditions
    def boundary_lb(x):
        return np.isclose(x[0], lb)
    
    def boundary_ub(x):
        return np.isclose(x[0], ub)
    
    # Define Dirichlet BCs
    u_lb = fem.Function(V)
    u_lb.interpolate(lambda x: np.full((x.shape[1],),0.0))
    bc_lb = fem.dirichletbc(u_lb, locate_dofs_geometrical(V, boundary_lb), V)
    
    u_ub = fem.Function(V)
    u_ub.interpolate(lambda x: np.full((x.shape[1],),0.0))
    bc_ub = fem.dirichletbc(u_ub, locate_dofs_geometrical(V, boundary_ub), V)
    
    bcs = [bc_lb, bc_ub]
    
    # Initial condition: max(cp * (x - K), 0) for call, max(cp * (K - x), 0) for put
    if cp == 1:
        initial_expr = lambda x: np.maximum(cp * (x[0] - K), 0)
    else:
        initial_expr = lambda x: np.maximum(cp * (K - x[0]), 0)
    
    u0 = fem.Function(V)
    u0.interpolate(initial_expr)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Coefficients in the PDE
    a_coeff = 0.5 * sigma**2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    b_coeff = ((r - q) * ufl.inner(ufl.grad(u), v)) * ufl.dx
    c_coeff = r * u * v * ufl.dx
    
    # Time-stepping parameters
    A = fem.form(u * v * ufl.dx + dt * (a_coeff + b_coeff + c_coeff))
    L_form = fem.form(u0 * v * ufl.dx)
    
    # Assemble the system matrix (only needs to be done once if A is time-independent)
    A_matrix = fem.petsc.assemble_matrix(A, bcs)
    A_matrix.assemble()
    
    # Create PETSc linear solver
    solver = PETSc.KSP().create(mesh_domain.comm)
    solver.setOperators(A_matrix)
    solver.setType(PETSc.KSP.Type.BJACOBI)
    solver.getPC().setType(PETSc.PC.Type.JACOBI)
    solver.setFromOptions()
    
    # Time-stepping loop
    u_new = fem.Function(V)
    for _ in range(int(T / dt)):
        # Assemble the RHS
        b = fem.petsc.assemble_vector(L_form)
        fem.petsc.apply_lifting(b, [a_coeff + b_coeff + c_coeff], bcs)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs, b)
        
        # Solve the linear system
        solver.solve(b, u_new.vector)
        u_new.x.scatter_forward()
        
        # Update for next time step
        u0.x.array[:] = u_new.x.array[:]
    
    # Interpolate to find the option price at s0
    s0_value = fem.Function(V)
    s0_value.interpolate(lambda x: s0)
    
    # Evaluate the solution at s0
    # Note: In 1D, you can find the closest node or use interpolation
    option_price = u_new(s0)[0]
    return option_price


############################################################################


# Heston stochastic volatility model (Analytical solution)
def dbo_heston_anly(s0=100., y0=0.12, K=85., r=0.03, kappa=1.5, theta=0.1, xi=0.5, cp=1, T=1., L=65., U=135., n_terms=300000):
    ans = 0.0
    for n in range(1, n_terms + 1):
        kn = pi * n / log(U / L)
        mu = -0.5 * kappa
        zeta = 0.5 * sqrt(kn**2 * xi**2 + kappa**2 + xi**2 / 4.)
        AA = -kappa * theta * (mu + zeta) * T - kappa * theta * log((-mu + zeta + (mu + zeta) * exp(-2. * zeta * T)) / (2. * zeta))
        BB = (xi ** 2 * (kn ** 2 + 0.25) * (1 - exp(-2. * zeta * T))) / (4. * (-mu + zeta + (mu + zeta) * exp(-2. * zeta * T)))
        if cp == 1:
            phin = 2. * ((-1)**(n+1) * kn * (sqrt(U / K) - sqrt(K / U)) + sin(kn * log(L / K))) / ((kn ** 2 + 0.25) * log(U / L))
        else:
            phin = 2. * (kn * (sqrt(K / L) - sqrt(L / K)) + sin(kn * log(L / K))) / ((kn ** 2 + 0.25) * log(U / L))
        ans += exp(2. * (AA - BB * y0) / (xi**2)) * phin * sin(kn * log(s0 / L))
    
    return exp(- r * T) * sqrt(s0 * K) * ans

# Finite Element Method (FEM) implementation using FEniCSx for Double Barrier Options under the Heston Model
def dbo_heston_fem(s0=100., y0=0.12, K=85., r=0.03, q=0.03, kappa=1.5, theta=0.1, xi=0.5, rho=0.0, cp=1, T=1., L=65., U=135., dt=1./100):
    # Transform variables
    s0 = log(s0 / K)
    
    # Define the computational domain
    mesh_size = (100, 100)
    s_min, s_max = log(L / K), log(U / K)
    y_min, y_max = 0.0, 3.0
    mesh_domain = mesh.create_rectangle(MPI.COMM_WORLD,
                                        [np.array([s_min, y_min]), np.array([s_max, y_max])],
                                        mesh_size,
                                        mesh.CellType.quadrilateral)
    
    # Define function space (quadratic Lagrange elements)
    V = fem.FunctionSpace(mesh_domain, ("Lagrange", 2))
    
    # Define boundary conditions
    def boundary_s_min(x):
        return np.isclose(x[0], s_min)
    
    def boundary_s_max(x):
        return np.isclose(x[0], s_max)
    
    # Boundary value expressions
    u_s_min = fem.Function(V)
    u_s_min.interpolate(lambda x: 0.0)
    bc_s_min = fem.dirichletbc(u_s_min, locate_dofs_geometrical(V, boundary_s_min), V)
    
    u_s_max = fem.Function(V)
    u_s_max.interpolate(lambda x: 0.0)
    bc_s_max = fem.dirichletbc(u_s_max, locate_dofs_geometrical(V, boundary_s_max), V)
    
    bcs = [bc_s_min, bc_s_max]
    
    # Initial condition: max(cp * (K * exp(x[0]) - K), 0) for call, etc.
    if cp == 1:
        initial_expr = lambda x: np.maximum(cp * (K * np.exp(x[0]) - K), 0)
    else:
        initial_expr = lambda x: np.maximum(cp * (K - K * np.exp(x[0])), 0)
    
    u0 = fem.Function(V)
    u0.interpolate(initial_expr)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Define coefficients for the PDE
    A_matrix = ufl.as_matrix([[y / 2, rho * xi * y / 2],
                              [rho * xi * y / 2, xi**2 * y / 2]])
    B_vector = ufl.as_vector([((y + rho * xi) / 2 - (r - q)),
                              (xi**2 / 2 - kappa * (theta - y))])
    
    # Define the PDE
    a = u * v * ufl.dx - ufl.dot(A_matrix @ ufl.grad(u), ufl.grad(v)) * dt * ufl.dx \
        + ufl.dot(B_vector, ufl.grad(u)) * v * dt * ufl.dx + r * u * v * dt * ufl.dx
    
    L_form = u0 * v * ufl.dx
    
    # Assemble the system matrix (time-dependent if coefficients depend on time)
    A = fem.form(a)
    A_matrix_petsc = fem.petsc.assemble_matrix(A, bcs)
    A_matrix_petsc.assemble()
    
    # Create PETSc linear solver
    solver = PETSc.KSP().create(mesh_domain.comm)
    solver.setOperators(A_matrix_petsc)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setFromOptions()
    
    # Time-stepping loop
    u_new = fem.Function(V)
    for _ in range(int(T / dt)):
        # Assemble RHS
        b = fem.petsc.assemble_vector(L_form)
        fem.petsc.apply_lifting(b, [a], bcs)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs, b)
        
        # Solve the linear system
        solver.solve(b, u_new.vector)
        u_new.x.scatter_forward()
        
        # Update for next time step
        u0.x.array[:] = u_new.x.array[:]
    
    # Evaluate the solution at (s0, y0)
    # Since FEniCSx does not provide a direct evaluation method, use interpolation or projection
    # Here, we'll use a simple nearest neighbor approach for demonstration
    
    # Create a point to evaluate
    point = np.array([s0, y0], dtype=np.float64)
    
    # Find the nearest node
    coordinates = mesh_domain.geometry.x
    distances = np.linalg.norm(coordinates - point, axis=1)
    nearest = np.argmin(distances)
    option_price = u_new.x.array[nearest]
    
    return option_price

# Example of computing option prices for multiple scenarios
def compute_multiple_option_prices():
    s0_values = linspace(80, 120, 5)  # Example range of initial stock prices
    K = 100
    sigma = 0.2
    r = 0.05
    T = 1
    lb = 50
    ub = 150
    dt = 1 / 100
    cp = 1  # Call option

    # Black-Scholes prices for multiple initial prices
    bs_prices = [dbo_bs_anly(sigma, r, T, L=lb, U=ub, S = s0, X=K, cp=cp) for s0 in s0_values]
    if MPI.COMM_WORLD.rank == 0:
        print("Black-Scholes prices:", bs_prices)

    # Heston prices for multiple initial prices
    heston_prices = [dbo_heston_anly(s0, y0=0.12, K=K, r=r, cp=cp, T=T, L=lb, U=ub) for s0 in s0_values]
    if MPI.COMM_WORLD.rank == 0:
        print("Heston prices:", heston_prices)

    # Finite Element Method for Double Barrier Options using Black-Scholes
    fem_bs_prices = [dbo_bs_fem(s0, K, sigma, r, T, dt, lb, ub, cp=cp) for s0 in s0_values]
    if MPI.COMM_WORLD.rank == 0:
        print("FEM Black-Scholes prices:", fem_bs_prices)

    # Finite Element Method for Double Barrier Options using Heston model
    fem_heston_prices = [dbo_heston_fem(s0, y0=0.12, K=K, r=r, cp=cp, T=T, L=lb, U=ub, dt=dt) for s0 in s0_values]
    if MPI.COMM_WORLD.rank == 0:
        print("FEM Heston prices:", fem_heston_prices)

if __name__ == "__main__":
    compute_multiple_option_prices()
