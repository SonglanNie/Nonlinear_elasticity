### Primal linear elasticity MMS

from firedrake import *
from firedrake.petsc import PETSc
import numpy
from tabulate import tabulate
import argparse

parser = argparse.ArgumentParser(add_help = False)
parser.add_argument("--mu", type = float, default = 1.0)
parser.add_argument("--nu", type = float, default = 0.25)
parser.add_argument("--uex", type = str, default = "[sin(pi*x)*sin(pi*y), sin(pi*x)*sin(pi*y)]")
parser.add_argument("--Mesh", type = str, default = "UnitSquareMesh(N_base, N_base, diagonal = 'crossed')")
parser.add_argument("--warp", type = str, default = "False")
parser.add_argument("--N_Newt", type = int, default = 4)
args, _ = parser.parse_known_args()

green = '\033[92m'
white = '\033[0m'
print(green + "nu =", args.nu, ", uex =", args.uex) # removed ,white for consistency

elt = "CG" # element type
deg = 4

N_Newt = args.N_Newt
N_base = 2 # number of cells in each direction on the coarsest mesh
mesh = eval(args.Mesh)
levels = 3 # number of meshes
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
mu = Constant(args.mu)
nu = Constant(args.nu)
lam = 2*args.mu*args.nu/(1 - 2*args.nu)

# For plane stress conditions:
#lam = 2*args.mu*lam/(lam + 2*args.mu)

print(green + "lam =", lam) # removed ,white for consistency
lam = Constant(lam)

I = Identity(2)

# L2 norm over the whole domain
def L2norm(f):
    return sqrt(assemble(inner(f, f)*dx))

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

errors = []
results = []
for i, msh in enumerate(mh):
    N = N_base * 2**i # The reciprocal of the current value of h
    x, y = SpatialCoordinate(msh)
    uex = as_vector(eval(args.uex))
    # print(l2norm(uex)**2)
    # Strong form residual of the MMS
    # resid = -div(sigma(uex)) # can weaken the def to allow less smooth uex
    V = VectorFunctionSpace(msh, elt, deg)
    u = Function(V).interpolate(uex)
    uex_int = Function(V).interpolate(uex)
    # u = Function(V)
    du = TrialFunction(V)
    v = TestFunction(V)
    du_sol = Function(V)
    bc = DirichletBC(V, Constant([0, 0]), "on_boundary")      
    n = FacetNormal(msh)
    # Overall residual, incorporating MMS residual and boundary residual
    params = {'ksp_type': 'preonly',
             'pc_type': 'lu'
            }
    for j in range(N_Newt):
        # linearised VP
        a = (inner(sigma(du), eps(v))*dx - inner(4*l2norm(u)**2*inner(u,du)*u + l2norm(u)**4*du, v)*dx)
        F = ( 
              inner(l2norm(u)**4*u, v)*dx
            - inner(sigma(u), eps(v))*dx
            )
        solve(a == F, du_sol, 
            bcs = bc
        #    solver_parameters = params
        )
        # du_sol = Function(V).interpolate(du_sol)
        u += du_sol
        # print(max(abs(u.vector().array())))
        print(max(abs(du_sol.vector().array())))
    # error estimate
    error_disp_L2 = L2norm(uex - u)
    error_disp_H1 = sqrt(error_disp_L2**2 + L2norm(grad(uex - u))**2)        
    h = 1 / N
    if i == 0:
        EOC_disp_L2, EOC_disp_H1 = 0, 0
    else:
        h_prev = 2*h
        N_prev = N/2
        prev_disp_L2 = errors[i - 1][1]
        prev_disp_H1 = errors[i - 1][3]
        EOC_disp_L2 = EOC(h, h_prev, error_disp_L2, prev_disp_L2)
        EOC_disp_H1 = EOC(h, h_prev, error_disp_H1, prev_disp_H1)
    errors.append([N, error_disp_L2, GREEN % (EOC_disp_L2), error_disp_H1, GREEN % (EOC_disp_H1)])

print("\n", tabulate(errors, headers = ['N', 'Disp error in L2', GREEN % ('Disp EOC in L2'), 'Disp error in H1', GREEN % ('Disp EOC in H1')]), "\n")

#Citations.print_at_exit()
