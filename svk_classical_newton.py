"""
Solves the SVK hyperelasticity problem with a Method of Manufactured
Solutions (MMS) using Firedrake's built-in classical Newton solver.

This script lets Firedrake to compute the Jacobian G_u (Eq 5.11) automatically.
"""

# Set OMP_NUM_THREADS=1 before importing Firedrake
import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from firedrake.petsc import PETSc
import numpy
from tabulate import tabulate
import argparse
from utils import L4norm, W14norm, EOC

I = Identity(2)  # 2x2 Identity matrix

# Setup Argument Parser
parser = argparse.ArgumentParser(description="SVK MMS Solver (Classical Newton)")
parser.add_argument("--mu", type=float, default=1.0)
parser.add_argument("--nu", type=float, default=0.25)
parser.add_argument("--uex", type=str, default="[cos(x)*x*(1 - x)*y*(1 - y), cos(y)*x*(1 - x)*y*(1 - y)]")
parser.add_argument("--deg", type=int, default=4)
parser.add_argument("--levels", type=int, default=5)
parser.add_argument("--N_base", type=int, default=2)
parser.add_argument("--elt", type=str, default="CG")
args, _ = parser.parse_known_args()

# Setup Parameters
mu = Constant(args.mu)
nu = Constant(args.nu)
lam = 2 * mu * nu / (1 - 2 * nu)
lam = Constant(lam)

print(f"--- SVK MMS (Classical Newton)")
print(f"mu = {args.mu}, nu = {args.nu} (lam = {float(lam):.4f})")
print(f"Element: {args.elt} (P{args.deg})")
print(f"Exact solution: u = {args.uex}\n")

# Setup Mesh Hierarchy
mesh = UnitSquareMesh(args.N_base, args.N_base, diagonal='crossed')
mh = MeshHierarchy(mesh, args.levels - 1)

# Solver Parameters (From svk.py)
# Use Newton Line Search (snes_type: newtonls)
solver_params = {
    "snes_type": "newtonls",
    "snes_max_it": 100,
    "snes_rtol": 1e-8,
    "snes_atol": 1e-10,
    "snes_stol": 0.0,
    "snes_monitor": None,
    "snes_converged_reason": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

errors = []
for i, msh in enumerate(mh):
    N = args.N_base * 2**i
    h = 1.0 / N

    # Define Function Spaces
    V = VectorFunctionSpace(msh, args.elt, args.deg)
    u = Function(V, name="Displacement")
    v = TestFunction(V)
    
    # Set initial guess (e.g., zero or half the solution)
    # u.interpolate(0.5 * uex) 
    u.assign(0.0)

    # Define MMS
    x, y = SpatialCoordinate(msh)
    uex = as_vector(eval(args.uex))
    
    # Deformation gradient for exact solution
    Fex = I + grad(uex)
    
    # Forcing term f = -div(P(uex)) (Report, Eq 5.8)
    # P(u) = mu*(F^T*F - I) + lam*(div(u) + 0.5*|grad(u)|^2)*F
    P_ex = mu * (dot(transpose(Fex), Fex) - I) + \
           lam * (div(uex) + 0.5 * inner(grad(uex), grad(uex))) * Fex
    f = -div(P_ex)

    # Define Variational Form (Report, Eq 5.8)
    # G(u, v) = inner(P(u), grad(v)) - inner(f, v) = 0
    F_u = I + grad(u)
    P_u = mu * (dot(transpose(F_u), F_u) - I) + \
          lam * (div(u) + 0.5 * inner(grad(u), grad(u))) * F_u
    
    G = inner(P_u, grad(v)) * dx - inner(f, v) * dx

    # Boundary conditions (u=0 on boundary)
    bc = DirichletBC(V, Constant([0, 0]), "on_boundary")

    # Solve Nonlinear System
    # Firedrake builds the Jacobian G_u (Eq 5.11) automatically
    # and runs the Newton iteration (Alg 1).
    print(f"--- Solving on mesh {i} (N={N})")
    solve(G == 0, u, bcs=bc, solver_parameters=solver_params)

    # Error Calculation
    error_L4 = L4norm(uex - u)
    error_W14 = W14norm(uex - u)

    if i == 0:
        eoc_L4, eoc_W14 = 0.0, 0.0
    else:
        h_prev = 2 * h
        prev_L4 = errors[i - 1][1]
        prev_W14 = errors[i - 1][3]
        eoc_L4 = EOC(h, h_prev, error_L4, prev_L4)
        eoc_W14 = EOC(h, h_prev, error_W14, prev_W14)

    errors.append([N, error_L4, f"{eoc_L4:.2f}", error_W14, f"{eoc_W14:.2f}"])

print("\n")
print(tabulate(errors, headers=['N', 'L4 Error', 'EOC', 'W1-4 Error', 'EOC'], tablefmt="pretty"))
print("\n")

