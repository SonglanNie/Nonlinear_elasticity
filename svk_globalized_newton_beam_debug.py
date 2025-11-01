"""
Solves the SVK beam problem with a traction load `g` on
the right boundary.

This script implements the globalisation methods from Section 6:
- Load Continuation (Algorithm 5)
- Backtracking Line Search (Algorithm 2)
- Critical Point Line Search (Algorithms 3 & 4)

The specific method is chosen using command-line arguments.
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
# Import from utils (Identity is redefined for clarity here)
from utils import L4norm, W14norm, EOC

I = Identity(2)  # 2x2 Identity matrix

# Setup Argument Parser
parser = argparse.ArgumentParser(description="SVK Beam (Globalised Newton)")
# Physics
parser.add_argument("--lam", type=float, default=1.0, help="First Lame parameter")
parser.add_argument("--mu", type=float, default=1.0, help="Second Lame parameter")
parser.add_argument("--kappa", type=float, default=0.1, help="Traction force magnitude")
# Discretization
# --- MODIFICATION: Use N_base to match old script ---
# parser.add_argument("--mesh_x", type=int, default=2, help="Base mesh cells in X")
# parser.add_argument("--mesh_y", type=int, default=2, help="Base mesh cells in Y")
parser.add_argument("--N_base", type=int, default=2, help="Base mesh cells (NxN), matches old script")
# --- END MODIFICATION ---
parser.add_argument("--levels", type=int, default=3, help="Number of mesh refinement levels")
parser.add_argument("--deg", type=int, default=4)
parser.add_argument("--elt", type=str, default="CG")
# Solver
parser.add_argument("--N_Newt", type=int, default=50, help="Max Newton iterations per step")
parser.add_argument("--tol", type=float, default=1e-8, help="Base Newton tolerance (now overridden by mesh-dependent tol)")
# Globalisation
parser.add_argument("--N_param", type=int, default=1, help="Number of load continuation steps (Alg 5)")
parser.add_argument("--N_line", type=int, default=5, help="Max line search iterations")
parser.add_argument("--LS_Type", type=int, default=0, help="Line Search Type: 0=None, 1=Backtrack (Alg 2), 2=CPLS-Quasi (Alg 3), 3=CPLS-Exact (Alg 4)")

args, _ = parser.parse_known_args()

# Setup Parameters
mu = Constant(args.mu)
lam = Constant(args.lam)
kappa = Constant(args.kappa)

# Traction force vector
g = as_vector([0, kappa])

print(f"--- SVK Beam (Globalised Newton)")
print(f"mu = {args.mu}, lam = {args.lam}, kappa = {args.kappa}")
# --- MODIFICATION: Update print statement ---
print(f"Mesh: {args.N_base}x{args.N_base} base cells, {args.levels} levels")
# --- END MODIFICATION ---
print(f"Solver: N_param={args.N_param}, LS_Type={args.LS_Type}, N_Newt={args.N_Newt}\n")

# Setup Mesh Hierarchy
# --- MODIFICATION: Use UnitSquareMesh to match old script exactly ---
# base_mesh = RectangleMesh(args.mesh_x, args.mesh_y, 1, 1, diagonal='crossed')
base_mesh = UnitSquareMesh(args.N_base, args.N_base, diagonal='crossed')
# --- END MODIFICATION ---
mh = MeshHierarchy(base_mesh, args.levels - 1)

# Solver Parameters (for the linear system)
linear_solver_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu'
}

# Complexity Counters
N_eval_total = 0  # Bilinear/Linear form evaluations
N_lin_total = 0   # Linear solves

# Loop over meshes
for i, msh in enumerate(mh):
    N_cells = msh.num_cells()
    print(f"--- Solving on mesh {i} (N_cells={N_cells})")

    # --- MODIFICATION: START ---
    # Define mesh-dependent tolerances, matching svk_beam_param_cont.py
    # Loose tolerance for intermediate load steps (line 313)
    loose_tol = 1e-2 / (20**i)
    # Strict tolerance for final load step (line 320)
    strict_tol = 5e-5 / (20**i)
    # --- MODIFICATION: END ---

    # Define Function Spaces
    V = VectorFunctionSpace(msh, args.elt, args.deg)
    u = Function(V, name="Displacement")
    v = TestFunction(V)
    du = TrialFunction(V)  # Newton update (delta_u^k)
    du_sol = Function(V, name="Newton Update")
    
    # Set initial guess
    u.assign(0.0)
    
    # Boundary condition (clamped on left side, x=0)
    # Boundary ID 1 is the left side
    bc = DirichletBC(V, Constant([0, 0]), 1)
    
    # Load parameter for continuation
    alpha = Constant(0.0)
    
    # Load Continuation Loop (Algorithm 5)
    load_steps = numpy.linspace(1.0 / args.N_param, 1.0, args.N_param)
    
    for step_num, load_val in enumerate(load_steps):
        alpha.assign(load_val)
        print(f"  Load Step {step_num + 1}/{args.N_param} (alpha = {load_val:.4f})")
        
        # --- MODIFICATION: START ---
        # Set the correct tolerance for this specific load step
        if step_num == len(load_steps) - 1:
            current_tol = strict_tol
            print(f"    Using strict tolerance: {current_tol:.2e}")
        else:
            current_tol = loose_tol
            print(f"    Using loose tolerance: {current_tol:.2e}")
        # --- MODIFICATION: END ---

        # Get initial residual norm for scaling
        F_u_init = I + grad(u)
        P_u_init = mu * (dot(transpose(F_u_init), F_u_init) - I) + \
                 lam * (div(u) + 0.5 * inner(grad(u), grad(u))) * F_u_init
        L_init = inner(P_u_init, grad(v)) * dx - inner(alpha * g, v) * ds(2)
        
        u_res_0 = max(numpy.linalg.norm(assemble(L_init, bcs=bc).dat.data_ro), 1e-16) # Avoid division by zero
        N_eval_total += 2 # For L_init and assemble

        # Manual Newton Loop (Algorithm 1)
        for j in range(args.N_Newt):
            
            # Define Residual G(u;v) = L(v) (Report, Eq 5.8)
            F_u = I + grad(u)
            P_u = mu * (dot(transpose(F_u), F_u) - I) + \
                  lam * (div(u) + 0.5 * inner(grad(u), grad(u))) * F_u
            
            # L = G(u;v) = inner(P(u), grad(v)) - inner(alpha*g, v)*ds(2)
            L = inner(P_u, grad(v)) * dx - inner(alpha * g, v) * ds(2)
            
            # Define Jacobian G_u(u;v,du) = a(du,v) (Report, Eq 5.11)
            term1 = inner(mu * (dot(transpose(grad(du)), F_u) + dot(transpose(F_u), grad(du))), grad(v)) * dx
            term2_1 = inner(lam * (div(du) + inner(grad(u), grad(du))) * F_u, grad(v)) * dx
            term2_2 = inner(lam * (div(u) + 0.5 * inner(grad(u), grad(u))) * grad(du), grad(v)) * dx
            a = term1 + term2_1 + term2_2
            
            # Assemble residual to get current scaled norm
            res_norm_current = numpy.linalg.norm(assemble(L, bcs=bc).dat.data_ro)
            res_norm_scaled = res_norm_current / u_res_0
            N_eval_total += 2 # 1 for 'a', 1 for 'L' (and assemble)
            
            # Find du s.t. a(du, v) = -L(v) (Alg 1, line 5)
            solve(a == -L, du_sol, bcs=bc, solver_parameters=linear_solver_params)
            N_lin_total += 1
            
            # Check Convergence (before line search)
            update_norm = W14norm(du_sol)

            # --- MODIFICATION: START ---
            # Use the robust check with the new MESH-DEPENDENT tolerance
            if update_norm < current_tol and res_norm_scaled < current_tol:
                print(f"    Iter {j}: Update W1,4 norm = {update_norm:.2e}, Scaled Res = {res_norm_scaled:.2e}. Converged.")
                break
            # --- MODIFICATION: END ---
            
            # Globalisation: Line Search (Alg 2-4)
            beta = 1.0  # Default step size
            
            if args.LS_Type == 1:
                # Algorithm 2: Backtracking Line Search
                # 'res_norm_current' is already computed above
                N_eval_total += 0 # Already counted
                
                for k in range(args.N_line):
                    u_new = u + beta * du_sol
                    
                    # Calculate new residual norm
                    F_new = I + grad(u_new)
                    P_new = mu * (dot(transpose(F_new), F_new) - I) + \
                            lam * (div(u_new) + 0.5 * inner(grad(u_new), grad(u_new))) * F_new
                    L_new = inner(P_new, grad(v)) * dx - inner(alpha * g, v) * ds(2)
                    res_norm_new = numpy.linalg.norm(assemble(L_new, bcs=bc).dat.data_ro)
                    N_eval_total += 2 # for L_new and assemble
                    
                    c = 1e-4  # Standard Armijo constant
                    if res_norm_new <= (1 - c * beta) * res_norm_current:
                        # Condition met
                        break
                    else:
                        beta *= 0.5  # Backtrack
                
            elif args.LS_Type == 2:
                # Algorithm 3: CPLS (Quasi-Newton / Secant)
                beta_vec = numpy.empty(args.N_line + 2)
                beta_vec[0] = 0.0
                beta_vec[1] = 1.0
                
                G_pre = assemble(action(L, du_sol)) # G(u;v) is L(v)
                N_eval_total += 1

                for k in range(args.N_line):
                    beta_cur = beta_vec[k + 1].item()
                    beta_pre = beta_vec[k].item()

                    u_new = u + beta_cur * du_sol
                    F_new = I + grad(u_new)
                    P_new = mu * (dot(transpose(F_new), F_new) - I) + \
                            lam * (div(u_new) + 0.5 * inner(grad(u_new), grad(u_new))) * F_new
                    L_new = inner(P_new, grad(v)) * dx - inner(alpha * g, v) * ds(2)
                    G_cur = assemble(action(L_new, du_sol))
                    N_eval_total += 2 # for L_new and action
                    
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
                    L_new = inner(P_new, grad(v)) * dx - inner(alpha * g, v) * ds(2)
                    G_cur = assemble(action(L_new, du_sol))
                    
                    # G_u(u + beta*du; du, du)
                    F_beta = I + grad(u_new)
                    term1_beta = inner(mu * (dot(transpose(grad(du)), F_beta) + dot(transpose(F_beta), grad(du))), grad(du)) * dx
                    term2_1_beta = inner(lam * (div(du) + inner(grad(u_new), grad(du))) * F_beta, grad(du)) * dx
                    term2_2_beta = inner(lam * (div(u_new) + 0.5 * inner(grad(u_new), grad(u_new))) * grad(du), grad(du)) * dx
                    a_beta = term1_beta + term2_1_beta + term2_2_beta
                    
                    Gu_cur = assemble(action(a_beta, du_sol))
                    
                    N_eval_total += 4 # 2 for L_new, 2 for a_beta
                    
                    if abs(Gu_cur) < 1e-12:
                        dbeta = 0.0
                    else:
                        dbeta = - G_cur / Gu_cur
                    
                    beta_vec[k+1] = beta_cur + dbeta
                    
                    if abs(dbeta) < 1e-5:
                        break
                
                beta = min(max(beta_vec[k+1].item(), 0.1), 2.0)

            # End of Line Search
            
            print(f"    Iter {j}: Update W1,4 norm = {update_norm:.2e}, Scaled Res = {res_norm_scaled:.2e}, Step size beta = {beta:.2f}")
            
            # Update u (Alg 1, line 9)
            u.assign(u + beta * du_sol)
            
        else:
            print(f"    WARNING: Newton failed to converge in {args.N_Newt} iterations.")

print("\n--- Simulation Complete")
print(f"Total Bilinear/Linear Form Evaluations (N_eval): {N_eval_total}")
print(f"Total Linear Solves (N_solve): {N_lin_total}")
print("\n")

# Save the final deformation
# File("output/final_u.pvd").write(u)

