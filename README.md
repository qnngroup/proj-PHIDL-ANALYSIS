# Toolkit for calculating number of squares from PHIDL geometry
Solves laplace equation in 2D for calculating current flow through a conductor given a fixed applied voltage and known material conductivity

# Installation
First build the package
```
$ python3 -m build
```

Next, install FEniCSx, following setup for [FEniCS/Dolfinx](https://github.com/FEniCS/dolfinx?tab=readme-ov-file#binary):
```
$ conda create -n <ENV_NAME> python=3.10
$ conda activate <ENV_NAME>
$ conda install -c conda-forge fenics-dolfinx mpich pyvista
```

Now install `gmsh` and python API
```
$ conda install -c conda-forge gmsh python-gmsh
```

Install the `phidlfem` package
```
$ pip install dist/phidlfem-<version>-py3-none-any.whl
```

# Usage

See `__main__` of `example.py`
