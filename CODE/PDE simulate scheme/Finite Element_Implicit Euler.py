
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.io import XDMFFile
import pyvista
from dolfinx.plot import vtk_mesh

# ------------------ Physical and Computational Parameters ------------------
# Define physical parameters (same as in your code)
gamma = 0.5
rho = 0.3
mu = 1
sigma_S = 0.2
K = 0.0
T = 1.0

alpha = 0.8
b = 0.05
kappa = 1.0

y_min = 0.0
y_max = 150.0
n_cells = 200
dt = 0.01
num_steps = int(T / dt)
correlation_factor = (1 - rho**2)

# Create a unit square mesh with triangular cells

nx, ny = 64, 64
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

V = fem.functionspace(domain, ("Lagrange", 1))  # Define FunctionSpace

# ------------------ Define Exact Solution ------------------
# Spatial coordinates using UFL
coords = ufl.SpatialCoordinate(domain)

class exact_solution:
    def __init__(self, mu, gamma, rho, sigma_S, t, n_terms):
        self.gamma = gamma
        self.rho = rho
        self.mu = mu
        self.sigma_S = sigma_S
        self.kappa = kappa
        self.n_terms = n_terms
        self.t = t  # Current time tau

    def __call__(self, x):
        """
        Exact solution:
        w(y, tau) = exp(gamma*(1 - rho^2)*g(y) + 0.5*gamma*sigma_S^2*tau - gamma*sigma_S*(W_T-W_t)))
        Where g(y) = max(y(T), 0)
        """
        # Assuming g(y) = max(y, 0)

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
        g_y = x[0]**2  # Apply max(y, 0)

        exponent = self.gamma * (1 - self.rho**2) * g_y - 0.5 * (self.mu**2/ self.sigma_S**2) * self.t - (self.mu / self.sigma_S)*np.random.normal(0, np.sqrt(self.t))
        return np.exp(exponent)

# Initialize the exact solution at time t=0
u_exact = exact_solution(mu = 1, gamma=0.5, rho=0.3, sigma_S=0.2, t=0)

# Define Dirichlet boundary condition using the exact solution
u_D = fem.Function(V)
u_D.interpolate(u_exact)

# Define boundary conditions (same as your original code)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

# ------------------ Set Up Time-Stepping Loop ------------------
# Initialize solution at the previous time step
u_n = fem.Function(V)
u_n.interpolate(u_exact)

f_value = 0  # Example constant source term
f = fem.Constant(domain, PETSc.ScalarType(f_value))

# Define trial and test functions
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Define weak form (same as your code)
alpha_const = fem.Constant(domain, PETSc.ScalarType(alpha))
b_const = fem.Constant(domain, PETSc.ScalarType(b))

F = (
    u * v * ufl.dx
    + dt * (
        0.5 * alpha_const**2 * ufl.dot(ufl.grad(u), ufl.grad(v))
        + (rho * mu / sigma_S * alpha_const - b_const) * ufl.grad(u)[0] * v
    ) * ufl.dx
    - (u_n + dt * f) * v * ufl.dx
)

a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

# Assemble the system matrix and solve (same as your code)
A = assemble_matrix(a, bcs=[bc])
A.assemble()

b = create_vector(L)
uh = fem.Function(V)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# XDMF output setup
with XDMFFile(domain.comm, "minimal2.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

    for n in range(num_steps):
        u_exact.t += dt  # Update the exact solution time
        u_D.interpolate(u_exact)

        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, L)

        apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc], b)

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        u_n.x.array[:] = uh.x.array[:]

        xdmf.write_function(uh, n*dt)

        if domain.comm.rank == 0 and n % (num_steps // 10) == 0:
            print(f"Time step {n+1}/{num_steps} completed.")

    # Optionally, write the final solution
    xdmf.write_function(uh, T)

# ------------------ Post-Processing ------------------
import pyvista as pv

# Compute L2 error
V_ex = fem.functionspace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)

error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute maximum error
error_max = domain.comm.allreduce(np.max(np.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")

# PyVista Plotting (same as your code)
if domain.comm.rank == 0:
    topology, cell_types, geometry = vtk_mesh(domain, domain.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    grid.point_data["uh"] = uh.x.array.real

    p = pv.Plotter(off_screen=True)
    p.add_mesh(grid, scalars="uh", cmap="viridis", show_edges=True)
    p.add_axes()
    p.view_xy()

    screenshot_path = "minimal2.png"
    p.show(screenshot=True, window_size=[800, 600], auto_close=True)
    p.screenshot(screenshot_path)
    p.close()
