import numpy as np
import json

from dolfinx import geometry

from phidlfem.utilities import make_mesh_2D
from phidlfem.solvers import solve_laplace


def get_squares(D, max_step=0.5):
    """
    Get number of squares in a two-terminal phidl device

    Parameters:
        D (Device): phidl Device
        max_step (Float): maximum step size for mesh refinement in microns

    Returns:
        sq (List[Float]): number of squares to ground from each port
    """
    D_hash = hash(D.hash_geometry() + max_step)
    try:
        # load hashes and check for a match
        with open(".phidlfem_D_hashes.json", "r") as f:
            hash_data = json.load(f)
        if D_hash in hash_data:
            return hash_data[D_hash]
    except OSError:
        hash_data = {}
        pass
    domain, facet_markers, n_dbc = make_mesh_2D(D, max_step)
    J, bb_tree = solve_laplace(domain, facet_markers, n_dbc, visualize=False)
    # get the current flow into/out of each port
    sq = np.zeros(len(D.get_ports()))
    currents = np.zeros(len(D.get_ports()))
    for p, port in enumerate(D.get_ports()):
        # port tangent
        tangent = port.endpoints[1] - port.endpoints[0]
        tangent = tangent / np.sqrt(np.sum(tangent**2))  # normalize
        # port normal
        normal = port.normal[1] - port.normal[0]
        normal = normal / np.sqrt(np.sum(normal**2))  # normalize
        endpoints = port.endpoints
        N = 501
        x = np.linspace(
            endpoints[0][0] - 0.2 * max_step * tangent[0] - 0.2 * max_step * normal[0],
            endpoints[1][0] + 0.2 * max_step * tangent[0] - 0.2 * max_step * normal[0],
            N,
        )
        y = np.linspace(
            endpoints[0][1] - 0.2 * max_step * tangent[1] - 0.2 * max_step * normal[1],
            endpoints[1][1] + 0.2 * max_step * tangent[1] - 0.2 * max_step * normal[1],
            N,
        )
        points = np.zeros((3, N))
        points[0] = x
        points[1] = y
        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        # Choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(
            domain, cell_candidates, points.T
        )
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        J_values = J.eval(points_on_proc, cells)
        dx = np.diff(points_on_proc[:, 0])
        dy = np.diff(points_on_proc[:, 1])
        # normal vector is dn = [-dy; dx]
        I = np.trapz(-dy * J_values[1:, 0] + dx * J_values[1:, 1])
        currents[p] = I
        sq[p] = abs(p / I)

    if abs(np.sum(currents)) > 1e-2:
        print(
            "WARNING, DEVICE CURRENTS DON'T SUM TO ZERO, SQUARE COUNT MAY BE INNACURATE"
        )
    hash_data[D_hash] = [s for s in sq]
    with open(".phidlfem_D_hashes.json", "w") as f:
        json.dump(hash_data, f)
    return np.abs(sq)


def visualize_laplace(D, max_step=0.5):
    domain, facet_markers, n_dbc = make_mesh_2D(D, max_step)
    solve_laplace(domain, facet_markers, n_dbc, visualize=True)
