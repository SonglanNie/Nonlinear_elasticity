"""
Solves the SVK hyperelasticity problem with a Method of Manufactured
Solutions (MMS) using the manual, globalised Newton solver.

This script is the counterpart to 'svk_beam_globalised.py' and
demonstrates the globalisation methods from Section 6 on the MMS problem.
It combines:
- Load Continuation (Algorithm 5) on the forcing term `f`
- Backtracking Line Search (Algorithm 2)
- Critical Point Line Search (Algorithms 3 & 4)
"""

# Set OMP_NUM_THREADS=1 before importing Firedrake
import os
os.environ["OMP_NUM_THREADS"] = "1"
# End of fix

from firedrake import *
from firedrake.petsc import PETSc
import numpy
from tabulate import tabulate
import argparse
from utils import L4norm, W14norm, EOC

I = Identity(2)  # 2x2 Identity matrix

# Setup Argument Parser
parser = argparse.ArgumentParser(description="SVK MMS (Globalised Newton)")
# Physics
parser.add_argument("--mu", type=float, default=1.0)
parser.add_argument("--nu", type=float, default=0.25)
parser.add_argument("--uex", type=str, default="[cos(x)*x*(1 - x)*y*(1 - y), cos(y)*x*(1 - x)*y*(1 - y)]")
# Discretization
parser.add_argument("--N_base", type=int, default=2, help="Base mesh cells (NxN)")
parser.add_argument("--levels", type=int, default=4, help="Number of mesh refinement levels")
parser.add_argument("--deg", type=int, default=4)
parser.add_argument("--elt", type=str, default="CG")
# Solver
parser.add_argument("--N_Newt", type=int, default=50, help="Max Newton iterations per step")
parser.add_argument("--tol", type=float, default=1e-8, help="Newton tolerance")
# Globalisation
parser.add_argument("--N_param", type=int, default=1, help="Number of load continuation steps (Alg 5)")
parser.add_argument("--N_line", type=int, default=5, help="Max line search iterations")
parser.add_argument("--LS_Type", type=int, default=0, help="Line Search Type: 0=None, 1=Backtrack (Alg 2), 2=CPLS-Quasi (Alg 3), 3=CPLS-Exact (Alg 4)")

args, _ = parser.parse_known_args()

# Setup Parameters
mu = Constant(args.mu)
nu = Constant(args.nu)
lam = 2 * mu * nu / (1 - 2 * nu)
lam = Constant(lam)

print(f"--- SVK MMS (Globalised Newton)")
print(f"mu = {args.mu}, nu = {args.nu} (lam = {float(lam):.4f})")
print(f"Element: {args.elt} (P{args.deg}), uex = {args.uex}")
print(f"Solver: N_param={args.N_param}, LS_Type={args.LS_Type}, N_Newt={args.N_Newt}\n")

# Setup Mesh Hierarchy
base_mesh = UnitSquareMesh(args.N_base, args.N_base, diagonal='crossed')
mh = MeshHierarchy(base_mesh, args.levels - 1)

# Solver Parameters (for the linear system)
linear_solver_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu'
}

# Complexity Counters
N_eval_total = 0  # Bilinear/Linear form evaluations
N_lin_total = 0   # Linear solves

errors = []
# Loop over meshes
for i, msh in enumerate(mh):
    N = args.N_base * 2**i
    h = 1.0 / N
    print(f"--- Solving on mesh {i} (N={N})")

    # Define Function Spaces
    V = VectorFunctionSpace(msh, args.elt, args.deg)
    u = Function(V, name="Displacement")
    v = TestFunction(V)
    du = TrialFunction(V)  # Newton update (delta_u^k)
    du_sol = Function(V, name="Newton Update")
    
    # Set initial guess
    u.assign(0.0)
    
    # Define MMS
    x, y = SpatialCoordinate(msh)
    uex_func = as_vector(eval(args.uex))
    
    # Deformation gradient for exact solution
    Fex = I + grad(uex_func)
    
    # Forcing term f = -div(P(uex)) (Report, Eq 5.8)
    P_ex = mu * (dot(transpose(Fex), Fex) - I) + \
           lam * (div(uex_func) + 0.5 * inner(grad(uex_func), grad(uex_func))) * Fex
    f = -div(P_ex)
    
    # Boundary condition (u=0 on all boundaries)
    bc = DirichletBC(V, Constant([0, 0]), "on_boundary")
    
    # Load parameter for continuation
    alpha = Constant(0.0)
    
    # Load Continuation Loop (Algorithm 5)
    load_steps = numpy.linspace(1.0 / args.N_param, 1.0, args.N_param)
    
    for step_num, load_val in enumerate(load_steps):
        alpha.assign(load_val)
        print(f"  Load Step {step_num + 1}/{args.N_param} (alpha = {load_val:.4f})")
        
        # Manual Newton Loop (Algorithm 1)
        for j in range(args.N_Newt):
            
            # Define Residual G(u;v) = L(v) (Report, Eq 5.8)
            # With MMS forcing f and load parameter alpha
            F_u = I + grad(u)
            P_u = mu * (dot(transpose(F_u), F_u) - I) + \
                  lam * (div(u) + 0.5 * inner(grad(u), grad(u))) * F_u
            
            # L = G(u;v) = inner(P(u), grad(v)) - inner(alpha*f, v)*dx
            L = inner(P_u, grad(v)) * dx - inner(alpha * f, v) * dx
            
            # Define Jacobian G_u(u;v,du) = a(du,v) (Report, Eq 5.11)
            term1 = inner(mu * (dot(transpose(grad(du)), F_u) + dot(transpose(F_u), grad(du))), grad(v)) * dx
            term2_1 = inner(lam * (div(du) + inner(grad(u), grad(du))) * F_u, grad(v)) * dx
            term2_2 = inner(lam * (div(u) + 0.5 * inner(grad(u), grad(u))) * grad(du), grad(v)) * dx
            a = term1 + term2_1 + term2_2
            
            # Count evaluations (1 for 'a', 1 for 'L')
            N_eval_total += 2
            
            # Find du s.t. a(du, v) = -L(v) (Alg 1, line 5)
            solve(a == -L, du_sol, bcs=bc, solver_parameters=linear_solver_params)
            N_lin_total += 1
            
            # Check Convergence (before line search)
            update_norm = W14norm(du_sol)
            if update_norm < args.tol:
                print(f"    Iter {j}: Update W1,4 norm = {update_norm:.2e}. Converged.")
                break
            
            # Globalisation: Line Search (Alg 2-4)
            beta = 1.0  # Default step size
            
            if args.LS_Type == 1:
                # Algorithm 2: Backtracking Line Search
                res_norm = numpy.linalg.norm(assemble(L, bcs=bc).dat.data_ro)
                
                N_eval_total += 1
                
                for k in range(args.N_line):
                    u_new = u + beta * du_sol
                    
                    F_new = I + grad(u_new)
                    P_new = mu * (dot(transpose(F_new), F_new) - I) + \
                            lam * (div(u_new) + 0.5 * inner(grad(u_new), grad(u_new))) * F_new
                    L_new = inner(P_new, grad(v)) * dx - inner(alpha * f, v) * dx

                    res_norm_new = numpy.linalg.norm(assemble(L_new, bcs=bc).dat.data_ro)

                    N_eval_total += 2 
                    c = 1e-4
                    # c = 0.99
                    # breakpoint()
                    if res_norm_new <= (1 - c * beta) * res_norm:
                        break
                    else:
                        beta *= 0.5
                
            elif args.LS_Type == 2:
                # Algorithm 3: CPLS (Quasi-Newton / Secant)
                beta_vec = numpy.empty(args.N_line + 2)
                beta_vec[0] = 0.0
                beta_vec[1] = 1.0
                G_pre = assemble(action(L, du_sol))
                N_eval_total += 1

                for k in range(args.N_line):
                    beta_cur = beta_vec[k + 1].item()
                    beta_pre = beta_vec[k].item()

                    u_new = u + beta_cur * du_sol
                    F_new = I + grad(u_new)
                    P_new = mu * (dot(transpose(F_new), F_new) - I) + \
                            lam * (div(u_new) + 0.5 * inner(grad(u_new), grad(u_new))) * F_new
                    L_new = inner(P_new, grad(v)) * dx - inner(alpha * f, v) * dx
                    G_cur = assemble(action(L_new, du_sol))
                    N_eval_total += 2
                    
                    if abs(G_cur - G_pre) < 1e-12:
                        beta_vec[k+2] = beta_cur
                        break
                    
                    dbeta = - (G_cur * (beta_cur - beta_pre)) / (G_cur - G_pre)
                    beta_vec[k+2] = beta_cur + dbeta
                    G_pre = G_cur
                    
                    if abs(dbeta) < 1e-5:
                        break
                
                beta = min(max(beta_vec[k+2].item(), 0.1), 2.0)

            elif args.LS_Type == 3:
                # Algorithm 4: CPLS (Exact Newton)
                beta_vec = numpy.empty(args.N_line + 1)
                beta_vec[0] = 1.0
                
                for k in range(args.N_line):
                    beta_cur = beta_vec[k].item()
                    
                    u_new = u + beta_cur * du_sol
                    
                    # G(u + beta*du; du)
                    F_new = I + grad(u_new)
                    P_new = mu * (dot(transpose(F_new), F_new) - I) + \
                            lam * (div(u_new) + 0.5 * inner(grad(u_new), grad(u_new))) * F_new
                    L_new = inner(P_new, grad(v)) * dx - inner(alpha * f, v) * dx
                    G_cur = assemble(action(L_new, du_sol))
                    
                    # G_u(u + beta*du; du, du)
                    F_beta = I + grad(u_new)
                    term1_beta = inner(mu * (dot(transpose(grad(du)), F_beta) + dot(transpose(F_beta), grad(du))), grad(du)) * dx
                    term2_1_beta = inner(lam * (div(du) + inner(grad(u_new), grad(du))) * F_beta, grad(du)) * dx
                    term2_2_beta = inner(lam * (div(u_new) + 0.5 * inner(grad(u_new), grad(u_new))) * grad(du), grad(du)) * dx
                    a_beta = term1_beta + term2_1_beta + term2_2_beta
                    
                    Gu_cur = assemble(action(a_beta, du_sol))
                    
                    N_eval_total += 4
                    
                    if abs(Gu_cur) < 1e-12:
                        dbeta = 0.0
                    else:
                        dbeta = - G_cur / Gu_cur
                    
                    beta_vec[k+1] = beta_cur + dbeta
                    
                    if abs(dbeta) < 1e-5:
                        break
                
                beta = min(max(beta_vec[k+1].item(), 0.1), 2.0)

            # End of Line Search
            
            print(f"    Iter {j}: Update W1,4 norm = {update_norm:.2e}, Step size beta = {beta:.2f}")
            
            u.assign(u + beta * du_sol)
            
        else:
            print(f"    WARNING: Newton failed to converge in {args.N_Newt} iterations.")

    # Error Calculation (after all loops for this mesh)
    error_L4 = L4norm(uex_func - u)
    error_W14 = W14norm(uex_func - u)

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

print("\n--- Simulation Complete")
print(f"Total Bilinear/Linear Form Evaluations (N_eval): {N_eval_total}")
print(f"Total Linear Solves (N_solve): {N_lin_total}")
print("\n")
