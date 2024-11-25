import pytest
import numpy as np

from poisson import solve_poisson, errornorm

def test_exact_solution():
    """
        Test that P2 elements recover the quadratic solution exactly up to rounding error.
    """
    uh, ue = solve_poisson(n=4, degree=2)
    error_H1 = errornorm(uh, ue, "H1")

    assert error_H1 < 1e-12
    
    

def test_convergence_P1():
    """Test that
    """
    pass

#python -m pytest