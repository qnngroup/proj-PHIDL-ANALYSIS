# Toolkit for analyzing PHIDL geometry with FEniCS/dolfinx
Solves laplace equation in 2D for calculating current flow through a conductor given a fixed applied voltage and known material conductivity.
This gives the number of squares in the conductor

# Installation
First build the package
```
$ python3 -m build
```

Now create an environment with the required dependencies (see [FEniCS/dolfinx README](https://github.com/FEniCS/dolfinx) for installation guidelines if you prefer to use a different virtual environment manager):
```
$ conda create -n <ENV_NAME> -f environment.yml
$ conda activate <ENV_NAME>
```

Finally install the `phidlfem` package that you built in the first step
```
$ pip install dist/phidlfem-<version>-py3-none-any.whl
```

# Usage

See `__main__` of `example.py`
