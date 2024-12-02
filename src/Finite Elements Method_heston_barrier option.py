import numpy as np
from numpy import sqrt, exp, log, pi, sin, linspace
import scipy.stats as st
from mpi4py import MPI
import ufl

from dolfinx import mesh, fem, plot
from dolfinx.fem import (Function, functionspace, locate_dofs_geometrical, 
                         dirichletbc, Constant, Expression, form)
import dolfinx.fem.petsc
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_interval, create_rectangle
from petsc4py import PETSc

# Define standard normal cumulative distribution function
N = st.norm.cdf

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
    V = fem.functionspace(mesh_domain, ("Lagrange", 2))
    
    # Define boundary conditions
    def boundary_s_min(x):
        return np.isclose(x[0], s_min)
    
    def boundary_s_max(x):
        return np.isclose(x[0], s_max)
    def initial_condition(x):
      return 
    # Boundary value expressions

    # Create a constant boundary value (e.g., 0.0 for Dirichlet BCs)
    u_s_min_value =  Function(V)
    u_s_min_value.interpolate(lambda x: x[0]*0)
    u_s_max_value = Function(V)
    u_s_max_value.interpolate(lambda x: x[0]*0)

    # Locate DOFs on the boundaries
    dofs_s_min = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], s_min))
    dofs_s_max = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], s_max))

    # Create Dirichlet boundary conditions
    bc_s_min = fem.dirichletbc(u_s_min_value, dofs_s_min)
    bc_s_max = fem.dirichletbc(u_s_max_value, dofs_s_max)

    # List of boundary conditions
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
    
   # Diffusion coefficients matrix (second-order terms)
    A_matrix = ufl.as_matrix([[0.5 * y0, 0.5 * rho * xi * y0],
                              [0.5 * rho * xi * y0, 0.5 * xi**2 * y0]])
    
    # Advection coefficients vector (first-order terms)
    B_vector = ufl.as_vector([
        (0.5 * (y0 + rho * xi) - (r - q)),
        (0.5 * xi**2 - kappa * (theta - y0))
    ])
    
    # Define the PDE's weak form (Variational Formulation)
    # Implicit Euler Time-Stepping: (u_new - u_old)/dt = L(u_new)
    # Rearranged: u_new - dt * L(u_new) = u_old
    # Variational Form: ∫u_new * v dx + dt * (Diffusion + Advection + Reaction terms) = ∫u_old * v dx
    
    a = u * v * ufl.dx  # Mass matrix term
    a += dt * (ufl.inner(A_matrix * ufl.grad(u), ufl.grad(v)) + 
              ufl.dot(B_vector, ufl.grad(u)) * v + 
              r * u * v) * ufl.dx  # Stiffness matrix and reaction terms

    # Initial condition expression: u0
    L_form = u0 * v * ufl.dx
    
    # Assemble the system matrix and vector
    a_fem = fem.form(a)
    A_matrix_petsc = fem.petsc.assemble_matrix(a_fem, bcs)
    A_matrix_petsc.assemble()
    
    # Assemble the right-hand side vector
    L_fem = fem.form(L_form)
    b = fem.petsc.assemble_vector(L_fem)
    fem.petsc.apply_lifting(b, [a_fem], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)
    
    # Create PETSc linear solver
    solver = PETSc.KSP().create(mesh_domain.comm)
    solver.setOperators(A_matrix_petsc)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setFromOptions()
    
    # Initialize solution functions
    u_new = fem.Function(V)
    
    # Time-stepping loop
    num_steps = int(T / dt)
    for step in range(num_steps):
        # Assemble the system matrix and RHS for the current time step
        A = fem.petsc.assemble_matrix(a_fem, bcs)
        A.assemble()
        
        # Assemble RHS vector
        b = fem.petsc.assemble_vector(L_fem)
        fem.petsc.apply_lifting(b, [a_fem], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs)
        
        # Solve the linear system: A * u_new = b
        solver.solve(b,u_new.x.petsc_vec)
        u_new.x.scatter_forward()
        
        # Update for the next time step
        u0.x.array[:] = u_new.x.array[:]
    
    # Evaluate the solution at (s0, y0)
    # Since FEniCSx does not provide a direct evaluation method, use interpolation or projection
    # Here, we'll use a simple nearest neighbor approach for demonstration
    
    # Create a point to evaluate
    point = np.array([s0, y0], dtype=np.float64)
    
    # Find the nearest node
    coordinates = mesh_domain.geometry.x[:, :2]
    distances = np.linalg.norm(coordinates - point, axis=1)
    nearest = np.argmin(distances)
    option_price = u_new.x.array[nearest]
    
    return option_price

analy_price = dbo_heston_anly(s0=100., y0=0.12, K=85., r=0.03, kappa=1.5, theta=0.1, xi=0.5, cp=1, T=1., L=65., U=135., n_terms=300000)
if MPI.COMM_WORLD.rank == 0:
    print("Analytical Heston prices:", analy_price)
fem_heston_prices = dbo_heston_fem(s0=100., y0=0.12, K=85., r=0.03, q=0, kappa=1.5, theta=0.1, xi=0.5, rho=0.5, cp=1, T=1., L=65., U=135., dt=1./100)
if MPI.COMM_WORLD.rank == 0:
    print("FEM Heston prices:", fem_heston_prices)
