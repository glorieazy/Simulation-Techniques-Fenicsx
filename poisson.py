from mpi4py import MPI
import numpy as np
import dolfinx
from dolfinx import mesh, fem, plot, io
from dolfinx.fem.petsc import LinearProblem
from ufl import SpatialCoordinate, TrialFunction, TestFunction, inner, grad, dx

def solve_poisson(n, degree):
    """Solves the Poisson-Dirichlet problem on the unit square with exact solution 1 + x^2 + y^2

    Args:
        

    Returns:

    """
    # Create mesh and define function space
    msh = mesh.create_unit_square(
    comm=MPI.COMM_WORLD,
    nx=n,
    ny=n
    )

    V = fem.functionspace(
    mesh=msh,
    element=("P", degree)
    )

    # Define boundary condition in topological way
    tdim = msh.topology.dim #topological dimension of the mesh msh
    fdim = tdim -1 #facet dimension
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=boundary_facets)
    
    def manufactured_solution(x):
        """ Evaluates a quadratic polynomial in x and y

        Args:
            x : Coordinates where x[0] represents the x-coordinates and x[1] the y-coordinates

        Returns:
            list: A solution for the x values from the equation.
        """
        return 1 + x[0]**2 + 2 * x[1]**2
    
    

    x = SpatialCoordinate(msh)
    ue = manufactured_solution(x)


    uD = fem.Function(V)
    uD.interpolate(manufactured_solution)

    bc = fem.dirichletbc(value=uD, dofs=boundary_dofs)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = fem.Constant(msh, -6.)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx

    # Compute solution
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    uh.name = "Solution u"

    return uh, ue

def errornorm(uh, ue, norm):
    """...

    Args:

    Returns:

    """
    if norm == "L2":
        L2form = fem.form((uh - ue)**2 * dx)
        L2error = np.sqrt(fem.assemble_scalar(L2form))
        #print("L2-error:", L2error)
        return L2error

    elif norm == "H1":
        H1form = fem.form((uh - ue)**2 * dx + inner(grad(uh - ue), grad(uh - ue)) * dx)
        H1error = np.sqrt(fem.assemble_scalar(H1form))
        #print("H1-error:", H1error)
        return H1error
    else:
        print("Norm parameter only in L2 or H1") 



def save_solution(uh):
    """...

    Args:

    Returns:

    """
    # Export the solution in VTX format
    msh=uh.function_space.mesh
    with io.VTXWriter(msh.comm, "results/poisson.bp", [uh]) as vtx:
        vtx.write(0.0)

# Main Code
if __name__ == "__main__":
    uh, ue = solve_poisson(
        n=4,
        degree=1
    )
    uh.function_space.mesh
    error_L2 = errornorm(uh, ue, "L2")
    error_H1 = errornorm(uh, ue, "H1")

    save_solution(uh)
