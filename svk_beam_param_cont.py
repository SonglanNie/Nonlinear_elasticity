### Primal linear elasticity MMS

from firedrake import *
from firedrake.petsc import PETSc
import numpy
from tabulate import tabulate
import argparse
# import copy # replaced by firedrake native copy function
import matplotlib.pyplot as plt
# from ufl import * # not sure why it immediately causes error even when nothing is changed

parser = argparse.ArgumentParser(add_help = False)
parser.add_argument("--lam", type = float, default = 1.0)
# parser.add_argument("--mu", type = float, default = 1.0)
# parser.add_argument("--nu", type = float, default = 0.25)
parser.add_argument("--kappa", type = float, default = 0.1)
parser.add_argument("--uex", type = str, default = "[cos(x)*x*(1 - x)*y*(1 - y), cos(y)*x*(1 - x)*y*(1 - y)]")
# parser.add_argument("--v_test", type = str, default = "[x*(1 - x)*y*(1 - y), x*(1 - x)*y*(1 - y)]")
# parser.add_argument("--Mesh", type = str, default = "RectangleMesh(4, 1, 4*N_base, N_base, diagonal = 'crossed')") # Union-jack-type mesh. Can use UnitSquareMesh(N_base, N_base) instead for regular mesh
parser.add_argument("--Mesh", type = str, default = "UnitSquareMesh(N_base, N_base, diagonal='crossed')")
parser.add_argument("--warp", type = str, default = "False")
parser.add_argument("--N_Newt", type = int, default = 50)
parser.add_argument("--N_param", type = int, default = 1)
parser.add_argument("--N_line", type = int, default = 3)
parser.add_argument("--LS_Type", type = int, default = 0)
parser.add_argument("--levels", type = int, default = 3)
args, _ = parser.parse_known_args()

green = '\033[92m'
white = '\033[0m'
# print(green + "nu =", args.nu, ", uex =", args.uex) # removed ,white for consistency
print(green + "lam =", args.lam, ", uex =", args.uex)

elt = "CG" # element type
deg = 4

# Complexity measure
N_eval = 0 # number of bilinear form and bounded linear functional evaluations
N_lin = 0 # number of linear solvers called

N_Newt = args.N_Newt # number of Newton's iterates
N_param = args.N_param # number of parameter continuation steps
N_line = args.N_line # number of line search attempted
LS_Type = args.LS_Type # line search type
N_base = 2 # number of cells in each direction on the coarsest mesh
mesh = eval(args.Mesh)
levels = args.levels # number of meshes
mh = MeshHierarchy(mesh, levels - 1) # a function giving a sequence of meshes obtained by refinement of the original, (levels - 1) times

if eval(args.warp):
    print("Warping mesh.")
    V = FunctionSpace(mesh, mesh.coordinates.ufl_element())
    eps = Constant(1 / 2**(N_base - 1))
    x, y = SpatialCoordinate(mesh)
    new = Function(V).interpolate(as_vector([x + eps*sin(2*pi*x)*sin(2*pi*y),
                                             y - eps*sin(2*pi*x)*sin(2*pi*y)]))
    coords = [new]
    for mesh in mh[1:]:
        fine = Function(mesh.coordinates.function_space())
        prolong(new, fine)
        coords.append(fine)
        new = fine
    for mesh, coord in zip(mh, coords):
        mesh.coordinates.assign(coord)

# Poisson ratio and first Lame constant; 
# second Lame constant
lam = Constant(args.lam)
mu = Constant(1.0)
# nu = Constant(args.nu)
# lam = 2*args.mu*args.nu/(1 - 2*args.nu)

# Neumann forcing term
kappa = Constant(args.kappa)
g = Constant([0, kappa])
# g = as_vector([0, kappa])

# For plane stress conditions:
#lam = 2*args.mu*lam/(lam + 2*args.mu)

# print(green + "lam =", lam) # removed ,white for consistency
I = Identity(2)

# L4 norm over the whole domain
def L4norm(f):
    return (assemble( ((inner(f, f))**2) *dx))**(1/4)

# W14 norm over the whole domain
def W14norm(f):
    return (L4norm(f)**4 + L4norm(grad(f))**4)**(1/4)

# l2 Euclidean norm for vectors
def l2norm(v):
    return inner(v,v)**(1/2)

# Isotropic elasticity tensor 
# elast_tens(eps) = sigma
def elast_tens(G):
    sig2 = 2*mu*G[0,1]
    sig3 = 2*mu*G[1,0]
    sig1 = (2*mu/(1 - 2*nu))*((1 - nu)*G[0,0] + nu*G[1,1])
    sig4 = (2*mu/(1 - 2*nu))*(nu*G[0,0] + (1 - nu)*G[1,1])
    return as_tensor([[sig1, sig2],
                      [sig3, sig4]])

# Linearised strain tensor
def eps(v):
    return sym(grad(v))

# (Equivalent to the above acting on eps(v))
def sigma(v):
    return 2.0*mu*eps(v) + lam*tr(eps(v))*I

# Experimental Order of Convergence
def EOC(h1, h2, E1, E2):
    return (numpy.log(E1) - numpy.log(E2))/(numpy.log(h1) - numpy.log(h2))

# # Residual definition
# def resid(uex):
#     Fex = I + grad(uex)
#     resid = -div(mu*(dot(transpose(Fex), Fex) - I) + lam*(div(uex) + 0.5*inner(grad(uex), grad(uex)))*Fex)
#     return resid

# Linear elast binear form
def a_lin(du, v):
    a = (inner(sigma(du), eps(v))*dx) # remove dx, should include du as an input object
    return a

# Linear elast bounded linear functional
def L_lin(u, v, g):
    L = (
          inner(g, v)*ds(2)
        - inner(sigma(u), eps(v))*dx
        )
    return L

# SVK bilinear form
def a_svk(u, du, v):
    F = I + grad(u)
    a = (inner(mu*(dot(transpose(F), grad(du)) + dot(transpose(grad(du)), F)) # corrected expression (du^T F not du F^T)
                   + lam*((div(du) + inner(grad(u), grad(du)))*F + (div(u) + 0.5*inner(grad(u), grad(u)))*grad(du)), grad(v))*dx )    
    return a

# SVK bounded linear functional
def L_svk(u, v, g):
    F = I + grad(u)
    L = ( 
          inner(g, v)*ds(2)
        - inner(mu*(dot(transpose(F), F) - I) + lam*(div(u) + 0.5*inner(grad(u), grad(u)))*F, grad(v))*dx 
        )    
    return L

# SVK residual
def G_svk(u, v, g): 
    #v = TestFunction(V)
    F = I + grad(u)
    G = (
          inner(mu*(dot(transpose(F), F) - I) + lam*(div(u) + 0.5*inner(grad(u), grad(u)))*F, grad(v))*dx 
        - inner(g, v)*ds(2)
        )
    #print(type(G))
    #G_assem = assemble(G, bcs=bc)
    #print(G_assem.vector().array())
    return G

# SVK derivative
def Gu_svk(u, du, v):
    return a_svk(u, du, v)

# Linear solver definition
def lin_solve(a, du_sol, L):
    params = {'ksp_type': 'preonly',
             'pc_type': 'lu'
            }
    solve(a == L, du_sol, 
              bcs = bc,
              solver_parameters = params
    )
    return du_sol

# svk Newton solver
def svk_Newt(L, u, du, v, g, N_Newt, tol, LS_Type, N_eval, N_lin, u_Newt = None):
    if u_Newt is not None:
        u_err = numpy.empty(N_Newt)
    G = G_svk(u, v, g)
    G_assem = assemble(G, bcs = bc) # BCs important 
    G_vector = G_assem.vector().array()
    u_res_0 = max(numpy.linalg.norm(G_vector), 2e-16)
    N_eval += 1      
    for j in range(N_Newt):
        # linearised VP
        # du_sol = Function(V)
        a = a_svk(u, du, v)
        lin_solve(a, du_sol, L)
        N_eval += 1
        N_lin += 1
        # du_sol = Function(V).interpolate(du_sol)
        # print(W14norm(du_sol))
        G = G_svk(u, v, g)
        G_assem = assemble(G, bcs = bc) # BCs important 
        G_vector = G_assem.vector().array()
        u_res = numpy.linalg.norm(G_vector)
        u_res_scaled = u_res/u_res_0
        print("scaled residual: " + str(u_res_scaled))
        # print("unscaled residual: " + str(u_res))
        N_eval += 1 
        if W14norm(du_sol) < tol and u_res_scaled < tol:
        # if W14norm(du_sol) < tol:
            print("Break Newton loop at index: " + str(j))
            if u_Newt is not None:
                u_err = u_err[0:j]
            break
        else:
            if LS_Type == 1:
            # Backtracking linesearch
                beta = 1
                for i in range(N_line):
                    # print("" + str(i) + " in N_line")
                    G = G_svk(u, v, g)
                    G_assem = assemble(G, bcs = bc) # BCs important 
                    G_vector = G_assem.vector().array()
                    u_norm = numpy.linalg.norm(G_vector) # residual norm
                    G_new = G_svk(u + beta*du_sol, v, g)
                    G_assem_new = assemble(G_new, bcs = bc)
                    G_vector_new = G_assem_new.vector().array()
                    u_norm_new = numpy.linalg.norm(G_vector_new)
                    N_eval += 2
                    if u_norm_new > (1 - 0.01*beta)*u_norm:
                        # (modified) Armijo condition: without the derivative of the residual.
                        # print(u_norm_new) # shorter step sizes give larger residual (possibly need to increase the step size?)
                        # print("True")
                        beta *= 0.5
                    else:
                        # print(u_norm_new)
                        # print("False")
                        break
            elif LS_Type == 2:
            # Critical point linesearch (Quasi-Newton)
                beta_vec = numpy.empty(N_line + 2)
                beta_vec[0] = 0
                beta_vec[1] = 1
                for i in range(N_line):
                    beta_cur = beta_vec[i + 1].item()
                    # print(beta_cur)
                    beta_pre= beta_vec[i].item()
                    if abs(beta_cur - beta_pre) < 1e-5:
                        beta_vec[-1] = beta_cur
                        # print(i)
                        break
                    # print(beta_pre)
                    G_cur = assemble(G_svk(u+beta_cur*du_sol, du_sol, g)) # BCs not needed
                    G_pre = assemble(G_svk(u+beta_pre*du_sol, du_sol, g))
                    N_eval += 2
                    # print(G_cur)
                    # print(G_pre)
                    dbeta = - (G_cur*(beta_cur - beta_pre))/(G_cur - G_pre)
                    # print(dbeta)
                    beta_vec[i+2] = beta_cur + dbeta
                # print("original beta: " + str(beta_vec))
                beta = min(max(beta_vec[-1].item(), 0.5), 3)
                # print("adjusted beta: " + str(beta))
            elif LS_Type == 3:
            # Critical point linesearch (Full-Newton)
                beta_vec = numpy.empty(N_line + 1)
                beta_vec[0] = 1
                for i in range(N_line):
                    beta_cur= beta_vec[i].item()
                    # print(beta_pre)
                    G_cur = assemble(G_svk(u+beta_cur*du_sol, du_sol, g)) # BCs not needed
                    Gu_cur = assemble(Gu_svk(u+beta_cur*du_sol, du_sol, du_sol))
                    N_eval += 2
                    # print(G_cur)
                    # print(G_pre)

                    dbeta = - G_cur/Gu_cur
                    # print(dbeta)
                    beta_vec[i+1] = beta_cur + dbeta
                    if abs(dbeta) < 1e-5:
                        beta_vec[-1] = beta_cur
                        # print(i)
                        break                    
                print("original beta: " + str(beta_vec))
                # beta = min(max(beta_vec[-1].item(), 0.5), 3) # regularised stepsize
                beta = beta_vec[-1].item()
                # print("(adjusted) final beta: " + str(beta))      
            else:
                beta = 1
        u += beta*du_sol
        # print("scaled residual: " + str(u_res/W14norm(u)))
        if u_Newt is not None:
            u_err[j] = W14norm(u - u_Newt)
            # print("error: " + str(u_err[j]))
        # print(W14norm(u - uex))
    if u_Newt is None:
        return u, N_eval, N_lin
    else:
        print(u_err)
        return u_err, N_eval, N_lin

# errors = []
# results = []
u_error = [None]*levels # computed errors for Newton iterates compared to the discrete solution
for i, msh in enumerate(mh):
    N = N_base * 2**i # The reciprocal of the current value of h
    x, y = SpatialCoordinate(msh)

    # print(l2norm(uex)**2)
    # Strong form residual of the MMS
    # 
    V = VectorFunctionSpace(msh, elt, deg)
    
    uex = as_vector(eval(args.uex)) # can try 0 as exact sol and small initial guess

    #uex = eval(args.uex) # can try 0 as exact sol and small initial guess
    # u = Function(V).interpolate(0.5*uex)
    u = Function(V)
    du = TrialFunction(V)
    v = TestFunction(V)
    du_sol = Function(V)
    bc = DirichletBC(V, Constant([0, 0]), 1)
    n = FacetNormal(msh)
    # tolerance
    tol = 1e-2/(20**i)
    # tol = 1e-2
    # Load continuation
    alpha = 1.0/N_param
    # v_test = Function(V).interpolate(as_vector(eval(args.v_test)))

    # Initial guess using linear elasticity
    # a_lin1 = a_lin(du, v)
    # L_lin1 = L_lin(u, v, alpha*g)
    # u = lin_solve(a_lin1, du_sol, L_lin1)
    # N_eval += 2
    # N_lin += 1

    # G = G_svk(u, v, g)
    # print(type(G))
    # G_assem = assemble(G, bcs=bc) # BCs important 
    # print(type(G_assem))
    # G_vector = G_assem.vector().array()
    # print(numpy.linalg.norm(G_vector))
    # print(l2norm(G_vector))
    # exit()
    # L_svk1 = L_svk(u, v, alpha*g)
    # svk_Newt(L_svk1, u, du, v, g, N_Newt, tol)
    # u *= 0.5
    # print(W14norm(u))

    # # plot meshes
    # fig, axes = plt.subplots()
    # triplot(msh, axes=axes)
    # axes.legend()
    # # fig.show()
    # plt.show()
    
    for k in range(N_param-1):  
        alpha_g = Constant(alpha * g)
        print("alpha * kappa: {0}".format(alpha_g))          
        L_svk1 = L_svk(u, v, alpha_g)
        N_eval += 1
        u, N_eval, N_lin = svk_Newt(L_svk1, u, du, v, alpha_g, N_Newt, tol, LS_Type, N_eval, N_lin)
        alpha += 1.0/N_param

    tol = 5e-5/(20**i) # use a high accuracy solver only for the last step
    # u_pre = copy.deepcopy(u)
    u_pre = u.copy(deepcopy=True)
    # print(str(k+1) + " N_param in " + str(i+1) + " mesh")
    L_svk1 = L_svk(u, v, g)
    N_eval += 1
    u, N_eval, N_lin = svk_Newt(L_svk1, u, du, v, g, N_Newt, tol, LS_Type, N_eval, N_lin)
    # print("Second run")
    # L_svk1 = L_svk(u_pre, v, g)
    # tol = 1e-6/(20**i)
    # u_error[i], N_eval, N_lin = svk_Newt(L_svk1, u_pre, du, v, g, N_Newt, tol, LS_Type, N_eval, N_lin, u_Newt=u)
            
    # print(alpha - 1/N_param)
    # # error estimate
    # error_disp_L4 = L4norm(uex - u)
    # error_disp_W14 = W14norm(uex - u)
    # h = 1 / N
    # if i == 0:
    #     EOC_disp_L4, EOC_disp_W14 = 0, 0
    # else:
    #     h_prev = 2*h
    #     N_prev = N/2
    #     prev_disp_L4 = errors[i - 1][1]
    #     prev_disp_W14 = errors[i - 1][3]
    #     EOC_disp_L4 = EOC(h, h_prev, error_disp_L4, prev_disp_L4)
    #     EOC_disp_W14 = EOC(h, h_prev, error_disp_W14, prev_disp_W14)
    # errors.append([N, error_disp_L4, GREEN % (EOC_disp_L4), error_disp_W14, GREEN % (EOC_disp_W14)])

# print("\n", tabulate(errors, headers = ['N', 'Disp error in L4', GREEN % ('Disp EOC in L4'), 'Disp error in W1-4', GREEN % ('Disp EOC in W1-4')]), "\n")
print("Number of bilinear form and bounded linear functional evaluations: " + str(N_eval))
print("Number of linear solvers called: " + str(N_lin))
# # #Citations.print_at_exit()

# Save the final deformation and (linearised) strain
# File("output/final_u_{kappa}.pvd".format(kappa = args.kappa)).write(u, project(sqrt(inner(eps(u), eps(u))), FunctionSpace(msh, "DG", 3)))

# # Plot |u^k_h - u_h|_W14 against number of iterates
# for i in range(levels):
#     x_data = numpy.array(range(len(u_error[i])))
#     y_data = numpy.log(u_error[i])
#     plt.scatter(x_data, y_data, c='r', label='data')
#     plt.show()

