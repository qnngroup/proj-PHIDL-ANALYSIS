from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner, as_vector, dot

import pyvista

# meshing
import gmsh
from dolfinx.io import gmshio

import phidl.geometry as pg
from phidl import quickplot as qp
from phidl import set_quickplot_options

import matplotlib.pyplot as plt

#set_quickplot_options(blocking=True, show_ports=True, new_window=True)

# tutorial on poisson (general case of Laplace):
# https://docs.fenicsproject.org/dolfinx/v0.7.3/python/demos/demo_poisson.html

# calculating resistance
# https://physics.stackexchange.com/questions/190355/calculating-the-resistance-of-a-3d-shape-between-two-points
# https://physics.stackexchange.com/questions/467205/resistance-of-an-object-with-arbitrary-shape

def make_mesh(D):
    """
    Make a mesh from phidl device

    Parameters:
        D (Device): phidl device

    Returns:

    """
    # info on meshing:
    # https://fenicsproject.discourse.group/t/using-facet-tags-to-define-boundary-conditions-gmsh-fenicsx-dolfinx/9408/3
    if len(D.get_ports()) != 2:
        raise ValueError("currently only support 2-port devices")
    
    # make the mesh
    gmsh.initialize()
    model = gmsh.model()
    
    # loop over all points and connect them with lines
    # if the line falls within a port (i.e. the current point and
    # previous point belong to the same port),
    # then put that line in the dirichlet boundary list,
    # otherwise put it in the neumann boundary list
    def closest_port(point, D):
        for p, port in enumerate(D.get_ports()):
            for endpoint in port.endpoints:
                if np.sum((point - endpoint)**2) < 1e-6:
                    return p
        return -1
    N_points = len(D.get_polygons()[0])
    prev_point = N_points - 1
    prev_port = closest_port(D.get_polygons()[0][prev_point], D)
    dirichlet_boundaries = [[]]
    # first add points
    for n,point in enumerate(D.get_polygons()[0]):
        model.geo.add_point(point[0], point[1], 0, 0, n)
    # then add edges
    for n,point in enumerate(D.get_polygons()[0]):
        cur_port = closest_port(point, D)
        line = model.geo.add_line(n, prev_point, n)
        if cur_port != -1:
            if cur_port != prev_port:
                dirichlet_boundaries.append([])
            else:
                dirichlet_boundaries[-1].append(line)
        prev_point = n
        prev_port = cur_port
    if len(dirichlet_boundaries[-1]) == 0:
        dirichlet_boundaries.pop()
    
    # create a loop that combines all curves just pass in
    # list(range(N_points)) since we labeled them 0...N_points-1
    model.geo.add_curve_loop(list(range(N_points)), 1)
    # create surface which spans the loop
    surf = model.geo.add_plane_surface([1], 1)
    # sync CAD model with new geometry
    DBC_ID = 1
    NBC_ID = 2
    gmsh.model.geo.synchronize()
    model.add_physical_group(2, [surf], 1)
    for b,boundary in enumerate(dirichlet_boundaries):
        model.add_physical_group(1, boundary, b)
    print(f'dirichlet_boundaries = {dirichlet_boundaries}')
    
    #model.add_physical_group(1, neumann_boundaries, NBC_ID)
    gmsh.option.set_number("Mesh.CharacteristicLengthMin", 0.01)
    gmsh.option.set_number("Mesh.CharacteristicLengthMax", 1)
    model.mesh.generate(2)
    
    gmsh_model_rank = 0
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_WORLD, gmsh_model_rank, gdim=2)
    return domain, cell_markers, facet_markers, dirichlet_boundaries

def solve_poisson(domain,
                  cell_markers,
                  facet_markers,
                  dirichlet_boundaries,
                  visualize=False):
    V = fem.FunctionSpace(domain, ("Lagrange", 1))
    
    bcs = []
    for b in range(len(dirichlet_boundaries)):
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

def get_squares(D):
    domain, cell_markers, facet_markers, dirichlet_boundaries = make_mesh(D)
    J, bb_tree = solve_poisson(domain, cell_markers, facet_markers, dirichlet_boundaries, visualize=False)
    # get the current flow into/out of each port
    currents = []
    for p, port in enumerate(D.get_ports()):
        # port tangent
        tangent = port.endpoints[1] - port.endpoints[0]
        tangent = tangent/np.sqrt(np.sum(tangent**2)) # normalize
        # port normal
        normal = port.normal[1] - port.normal[0]
        normal = normal/np.sqrt(np.sum(normal**2)) # normalize
        endpoints = port.endpoints
        N = 501
        x = np.linspace(endpoints[0][0] - 0.2*tangent[0], endpoints[1][0] + 0.2*tangent[0], N)
        y = np.linspace(endpoints[0][1] - 0.2*tangent[1], endpoints[1][1] + 0.2*tangent[1], N)
        points = np.zeros((3, N))
        points[0] = x
        points[1] = y
        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        # Choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        J_values = J.eval(points_on_proc, cells)
        dx = np.diff(points_on_proc[:,0])
        dy = np.diff(points_on_proc[:,1])
        # normal vector is dn = [-dy; dx]
        I = np.trapz(-dy*J_values[1:,0] + dx*J_values[1:,1])
        currents.append(I)
    
    if abs(np.sum(currents)) > 1e-2:
        print("WARNING, DEVICE CURRENTS DON'T SUM TO ZERO, SQUARE COUNT MAY BE INNACURATE")

    gmsh.finalize()
    return abs(1/currents[-1])

def visualize_poisson(D):
    domain, cell_markers, facet_markers, dirichlet_boundaries = make_mesh(D)
    solve_poisson(domain, cell_markers, facet_markers, dirichlet_boundaries, visualize=True)
