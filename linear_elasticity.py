"""
Solves the Linear Elasticity problem with a Method of Manufactured Solutions (MMS).

For a linear problem, Firedrake's `solve(F == 0, ...)` is equivalent to
a single Newton step.

"""

import os
import argparse

# Fix 1: Set OMP_NUM_THREADS before Firedrake is imported.
os.environ["OMP_NUM_THREADS"] = "1"

# Setup Argument Parser
# parser = argparse.ArgumentParser(description="Linear Elasticity MMS Solver", add_help=False)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--mu", type=float, default=1.0)
parser.add_argument("--nu", type=float, default=0.25)
parser.add_argument("--uex", type=str, default="[cos(x)*x*(1 - x)*y*(1 - y), cos(y)*x*(1 - x)*y*(1 - y)]")
parser.add_argument("--deg", type=int, default=4)
parser.add_argument("--levels", type=int, default=5)
parser.add_argument("--N_base", type=int, default=2)
parser.add_argument("--elt", type=str, default="CG")
args, _ = parser.parse_known_args()

from firedrake import *
from firedrake.petsc import PETSc
from tabulate import tabulate
from utils import eps, sigma, L2norm, EOC  # Import from utils

# Setup Parameters
mu = Constant(args.mu)
nu = Constant(args.nu)
lam = 2 * mu * nu / (1 - 2 * nu)
lam = Constant(lam)

print(f"--- Linear Elasticity MMS")
print(f"mu = {args.mu}, nu = {args.nu} (lam = {float(lam):.4f})")
print(f"Element: {args.elt} (P{args.deg})")
print(f"Exact solution: u = {args.uex}\n")

# Setup Mesh Hierarchy
mesh = UnitSquareMesh(args.N_base, args.N_base, diagonal='crossed')
mh = MeshHierarchy(mesh, args.levels - 1)

errors = []
for i, msh in enumerate(mh):
    N = args.N_base * 2**i
    h = 1.0 / N

    # Define Function Spaces and Solution
    V = VectorFunctionSpace(msh, args.elt, args.deg)
    u = Function(V, name="Displacement")
    v = TestFunction(V)

    # Define MMS
    x, y = SpatialCoordinate(msh)
    uex = as_vector(eval(args.uex))

    # Forcing term f = -div(sigma(uex)) (Report, Eq 2.4a)
    f = -div(sigma(uex, mu, lam))

    # Define Variational Form (Report, Eq 2.3)
    # a(u, v) = L(v)
    # We write this as F(u, v) = a(u, v) - L(v) = 0
    a = inner(sigma(u, mu, lam), eps(v)) * dx
    L = inner(f, v) * dx
    F = a - L

    # Boundary conditions (u=0 on boundary)
    bc = DirichletBC(V, Constant([0, 0]), "on_boundary")

    # Solve Linear System
    # Firedrake recognizes this is a linear problem and uses a linear solver.
    # This is equivalent to one Newton step (Report, Sec 5.1).
    solver_params = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    solve(F == 0, u, bcs=bc, solver_parameters=solver_params)

    # Error Calculation
    error_L2 = L2norm(uex - u)
    error_H1 = sqrt(error_L2**2 + L2norm(grad(uex - u))**2)

    if i == 0:
        eoc_L2, eoc_H1 = 0.0, 0.0
    else:
        h_prev = 2 * h
        prev_L2 = errors[i - 1][1]
        prev_H1 = errors[i - 1][3]
        eoc_L2 = EOC(h, h_prev, error_L2, prev_L2)
        eoc_H1 = EOC(h, h_prev, error_H1, prev_H1)

    errors.append([N, error_L2, f"{eoc_L2:.2f}", error_H1, f"{eoc_H1:.2f}"])

print(tabulate(errors, headers=['N', 'L2 Error', 'EOC', 'H1 Error', 'EOC'], tablefmt="pretty"))
print("\n")