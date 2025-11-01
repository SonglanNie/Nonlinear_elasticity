"""
Utility functions for Firedrake elasticity examples.
Contains common definitions for norms, tensors, and EOC calculations.
"""

from firedrake import *
import numpy

# Tensors and Constants

I = Identity(2)  # 2D Identity tensor

def eps(v):
    """
    Linearised strain tensor (symmetric gradient).
    From Report, Sec 2.1.
    """
    return sym(grad(v))

def sigma(v, mu, lam):
    """
    Linearised stress tensor (Hooke's Law).
    From Report, Sec 5.1 (derived from Eq 2.3).
    """
    return 2.0 * mu * eps(v) + lam * tr(eps(v)) * I

# Norm Definitions

def L2norm(f):
    """L2 norm over the whole domain."""
    return sqrt(assemble(inner(f, f) * dx))

def L4norm(f):
    """L4 norm over the whole domain."""
    return (assemble(((inner(f, f))**2) * dx))**(1/4)

def W14norm(f):
    """W1,4 norm over the whole domain."""
    return (L4norm(f)**4 + L4norm(grad(f))**4)**(1/4)

# EOC Calculation

def EOC(h1, h2, E1, E2):
    """
    Computes the Experimental Order of Convergence (EOC).
    """
    return (numpy.log(E1) - numpy.log(E2)) / (numpy.log(h1) - numpy.log(h2))
