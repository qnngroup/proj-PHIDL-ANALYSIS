from mpi4py import MPI
import numpy as np

# meshing
import gmsh
from dolfinx.io import gmshio

def make_mesh_2D(D, max_step):
    """
    Make a mesh from phidl device

    Parameters:
        D (Device): phidl device
        max_step (Float): maximum step size in microns

    Returns:
        domain (Mesh): Dolfinx mesh object
        facet_markers (MeshTags): Dolfinx mesh tags object
        n_dbc (Int): number of ports
    """
    # info on meshing:
    # https://fenicsproject.discourse.group/t/using-facet-tags-to-define-boundary-conditions-gmsh-fenicsx-dolfinx/9408/3
    
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
                if np.sum((point - endpoint)**2) < max_step/10:
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
    
    gmsh.option.set_number("Mesh.CharacteristicLengthMin", max_step/10)
    gmsh.option.set_number("Mesh.CharacteristicLengthMax", max_step)
    model.mesh.generate(2)
    
    gmsh_model_rank = 0
    domain, _, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_WORLD, gmsh_model_rank, gdim=2)
    gmsh.finalize()
    return domain, facet_markers, len(dirichlet_boundaries)

