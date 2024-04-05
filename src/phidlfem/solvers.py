import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot, geometry
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner, as_vector, dot
from petsc4py.PETSc import ScalarType

import pyvista

# tutorial on poisson (general case of Laplace):
# https://docs.fenicsproject.org/dolfinx/v0.7.3/python/demos/demo_poisson.html

# calculating resistance
# https://physics.stackexchange.com/questions/190355/calculating-the-resistance-of-a-3d-shape-between-two-points
# https://physics.stackexchange.com/questions/467205/resistance-of-an-object-with-arbitrary-shape

def solve_laplace(domain,
                  facet_markers,
                  n_dbc,
                  visualize=False):
    """
    Solve poisson equation on the specified mesh

    Parameters:
        domain (Mesh): domain over which to solve ODE
        facet_markers (MeshTags): list of 1D boundaries
        n_dbc (Int): number of dirichlet boundary conditions
        visualize (bool): if True, use pyvista to visualize solutions

    Returns:
        J (Function): current density
        bb_tree (BoundingBoxTree)
    """
    V = fem.FunctionSpace(domain, ("Lagrange", 1))
    
    bcs = []
    for b in range(n_dbc):
        dbc_entities = facet_markers.indices[facet_markers.values == b]
        dbc_dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=dbc_entities)
        bcs.append(fem.dirichletbc(value=ScalarType(b), dofs=dbc_dofs, V=V))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    a = inner(grad(u), grad(v)) * dx
    f = fem.Constant(domain, ScalarType(0))
    L = f * v * dx
    
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
   
    W = fem.FunctionSpace(domain, ufl.VectorElement("DG", domain.ufl_cell(), 0))
    U = fem.FunctionSpace(domain, ("Lagrange", 1))
    J = fem.Function(W)
    J_expr = fem.Expression(as_vector((-uh.dx(0), -uh.dx(1))), W.element.interpolation_points())
    J.interpolate(J_expr)
    J_mag = fem.Function(U)
    J_mag_expr = fem.Expression((uh.dx(0)**2 + uh.dx(1)**2)**0.5, U.element.interpolation_points())
    J_mag.interpolate(J_mag_expr)
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    if visualize:
        cells, types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["uh"] = uh.x.array.real
        grid.set_active_scalars("uh")
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)
        plotter.show_bounds()
        plotter.show()
        
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["J_mag"] = J_mag.x.array.real
        grid.set_active_scalars("J_mag")
        top_imap = domain.topology.index_map(domain.topology.dim)
        num_cells = top_imap.size_local + top_imap.num_ghosts
        midpoints = mesh.compute_midpoints(domain, domain.topology.dim, range(num_cells))
        
        num_dofs = W.dofmap.index_map.size_local + W.dofmap.index_map.num_ghosts
        assert (num_cells == num_dofs)
        values = np.zeros((num_dofs, 3), dtype=np.float64)
        values[:, :domain.geometry.dim] = J.x.array.real.reshape(num_dofs, W.dofmap.index_map_bs)
        plotter = pyvista.Plotter()
        cloud = pyvista.PolyData(midpoints)
        scale = 1/np.sqrt(np.abs(values[:,0]**2) + np.abs(values[:,1]**2))
        values[:,0] *= scale
        values[:,1] *= scale
        cloud["J"] = values
        glyphs = cloud.glyph("J")
        plotter.add_mesh(grid, show_edges=False)
        plotter.add_mesh(glyphs, color="white", show_scalar_bar=False)
        plotter.show()
    return J, bb_tree
