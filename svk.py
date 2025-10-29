### SVK hyperelasticity MMS

from firedrake import *
from firedrake.petsc import PETSc
import numpy
from tabulate import tabulate
import argparse

parser = argparse.ArgumentParser(add_help = False)
parser.add_argument("--mu", type = float, default = 1.0)
parser.add_argument("--nu", type = float, default = 0.25)
parser.add_argument("--uex", type = str, default = "[cos(x)*x*(1 - x)*y*(1 - y), cos(y)*x*(1 - x)*y*(1 - y)]")
parser.add_argument("--Mesh", type = str, default = "UnitSquareMesh(N_base, N_base, diagonal = 'crossed')")
parser.add_argument("--warp", type = str, default = "False")
args, _ = parser.parse_known_args()

green = '\033[92m'
white = '\033[0m'
print(green + "nu =", args.nu, ", uex =", args.uex, white)

elt = "CG"
deg = 4

N_base = 2
mesh = eval(args.Mesh)
levels = 4
mh = MeshHierarchy(mesh, levels - 1)
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
mu = Constant(args.mu)
nu = Constant(args.nu)
lam = 2*args.mu*args.nu/(1 - 2*args.nu)

# For plane stress conditions:
#lam = 2*args.mu*lam/(lam + 2*args.mu)

print(green + "lam =", lam, white)
lam = Constant(lam)

I = Identity(2)

# L4 norm over the whole domain
def L4norm(f):
    return (assemble( ((inner(f, f))**2) *dx))**(1/4)

# Isotropic elasticity tensor
def elast_tens(G):
    sig2 = 2*mu*G[0,1]
    sig3 = 2*mu*G[1,0]
    sig1 = (2*mu/(1 - 2*nu))*((1 - nu)*G[0,0] + nu*G[1,1])
    sig4 = (2*mu/(1 - 2*nu))*(nu*G[0,0] + (1 - nu)*G[1,1])
    return as_tensor([[sig1, sig2],
                      [sig3, sig4]])

# (Equivalent to the above acting on eps(v))
def sigma(v):
    return 2.0*mu*eps(v) + lam*tr(eps(v))*I

# Linearised strain tensor
def eps(v):
    return sym(grad(v))

# Experimental Order of Convergence
def EOC(h1, h2, E1, E2):
    return (numpy.log(E1) - numpy.log(E2))/(numpy.log(h1) - numpy.log(h2))

errors = []
results = []
for i, msh in enumerate(mh):
    N = N_base * 2**i
    x, y = SpatialCoordinate(msh)
    uex = as_vector(eval(args.uex))
    # Strong form residual of the MMS
    Fex = I + grad(uex)
    resid = -div(mu*(dot(transpose(Fex), Fex) - I) + lam*(div(uex) + 0.5*inner(grad(uex), grad(uex)))*Fex)
    V = VectorFunctionSpace(msh, elt, deg)
    # u = Function(V).interpolate(0.5*uex)
    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant([0, 0]), "on_boundary")
    n = FacetNormal(msh)
    F = I + grad(u)
    # Overall residual, incorporating MMS residual and boundary residual
    G = (
          inner(mu*(dot(transpose(F), F) - I) + lam*(div(u) + 0.5*inner(grad(u), grad(u)))*F, grad(v))*dx 
        - inner(resid, v)*dx
        #- inner(dot(sigma(uex), n), v)*ds
        )
    params = {
              "snes_type": "newtonls",
              #"snes_type": "ksponly",
              #"snes_linesearch_type": "basic",
              "snes_max_it": 100,
              "ksp_type": "preonly",
              "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps",
              "mat_type": "aij",
              "snes_rtol": 1e-5,
              "snes_atol": 1e-25,
              "snes_stol": 0.0,
              'snes_monitor': None,
              #'snes_view': None,
              #'ksp_monitor_true_residual': None,
              'snes_converged_reason': None,
              #'ksp_converged_reason': None
            }
    # lu = {
    #   "snes_type": "ksponly",
    #   "snes_monitor_cancel": None,
    #   "ksp_type": "preonly",
    #   "pc_type": "lu",
    #   "pc_factor_mat_solver_type": "mumps",
    #   "mat_mumps_icntl_14": 200,
    #   "mat_type": "aij"
    #  }
    solve(G == 0, u, 
        bcs = bc,
        solver_parameters = params)
    error_disp_L4 = L4norm(uex - u)
    error_disp_W14 = (error_disp_L4**4 + L4norm(grad(uex - u))**4)**(1/4)        
    h = 1 / N
    if i == 0:
        EOC_disp_L4, EOC_disp_W14 = 0, 0
    else:
        h_prev = 2*h
        N_prev = N/2
        prev_disp_L4 = errors[i - 1][1]
        prev_disp_W14 = errors[i - 1][3]
        EOC_disp_L4 = EOC(h, h_prev, error_disp_L4, prev_disp_L4)
        EOC_disp_W14 = EOC(h, h_prev, error_disp_W14, prev_disp_W14)
    errors.append([N, error_disp_L4, GREEN % (EOC_disp_L4), error_disp_W14, GREEN % (EOC_disp_W14)])

print("\n", tabulate(errors, headers = ['N', 'Disp error in L4', GREEN % ('Disp EOC in L4'), 'Disp error in W1-4', GREEN % ('Disp EOC in W1-4')]), "\n")

#Citations.print_at_exit()
