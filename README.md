# Ragged

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- [![Documentation Status][rtd-badge]][rtd-link] -->
<!-- [![Conda-Forge][conda-badge]][conda-link] -->

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/jpivarski/ragged/workflows/CI/badge.svg
[actions-link]:             https://github.com/jpivarski/ragged/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/ragged
[conda-link]:               https://github.com/conda-forge/ragged-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/jpivarski/ragged/discussions
[pypi-link]:                https://pypi.org/project/ragged/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/ragged
[pypi-version]:             https://img.shields.io/pypi/v/ragged
[rtd-badge]:                https://readthedocs.org/projects/ragged/badge/?version=latest
[rtd-link]:                 https://ragged.readthedocs.io/en/latest/?badge=latest
<!-- prettier-ignore-end -->

## Introduction

**Ragged** is a library for manipulating ragged arrays as though they were
**NumPy** or **CuPy** arrays, following the
[Array API specification](https://data-apis.org/array-api/latest/API_specification).

For example, this is a
[ragged/jagged array](https://en.wikipedia.org/wiki/Jagged_array):

```python
>>> import ragged
>>> a = ragged.array([[[1.1, 2.2, 3.3], []], [[4.4]], [], [[5.5, 6.6, 7.7, 8.8], [9.9]]])
>>> a
ragged.array([
    [[1.1, 2.2, 3.3], []],
    [[4.4]],
    [],
    [[5.5, 6.6, 7.7, 8.8], [9.9]]
])
```

The values are all floating-point numbers, so `a.dtype` is `float64`,

```python
>>> a.dtype
dtype('float64')
```

but `a.shape` has non-integer dimensions to account for the fact that some of
its list lengths are non-uniform:

```python
>>> a.shape
(4, None, None)
```

In general, a `ragged.array` can have any mixture of regular and irregular
dimensions, though `shape[0]` (the length) is always an integer. This convention
follows the **Array API**'s specification for
[array.shape](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.shape.html#array_api.array.shape),
which must be a tuple of `int` or `None`:

```python
array.shape: Tuple[Optional[int], ...]
```

(Our use of `None` to indicate a dimension without a single-valued size differs
from the **Array API**'s intention of specifying dimensions of _unknown_ size,
but it follows the technical specification. **Array API**-consuming libraries
can try using **Ragged** to find out if they are ragged-ready.)

All of the normal elementwise and reducing functions apply, as well as slices:

```python
>>> ragged.sqrt(a)
ragged.array([
    [[1.05, 1.48, 1.82], []],
    [[2.1]],
    [],
    [[2.35, 2.57, 2.77, 2.97], [3.15]]
])

>>> ragged.sum(a, axis=0)
ragged.array([
    [11, 8.8, 11, 8.8],
    [9.9]
])

>>> ragged.sum(a, axis=-1)
ragged.array([
    [6.6, 0],
    [4.4],
    [],
    [28.6, 9.9]
])

>>> a[-1, 0, 2]
ragged.array(7.7)

>>> a[a * 10 % 2 == 0]
ragged.array([
    [[2.2], []],
    [[4.4]],
    [],
    [[6.6, 8.8], []]
])
```

All of the methods, attributes, and functions in the **Array API** will be
implemented for **Ragged**, as well as conveniences that are not required by the
**Array API**. See
[open issues marked "todo"](https://github.com/jpivarski/ragged/issues?q=is%3Aissue+is%3Aopen+label%3Atodo)
for **Array API** functions that still need to be written (out of 120 in total).

**Ragged** has two `device` values, `"cpu"` (backed by **NumPy**) and `"cuda"`
(backed by **CuPy**). Eventually, all operations will be identical for CPU and
GPU.

## Implementation

**Ragged** is implemented using **Awkward Array**
([code](https://github.com/scikit-hep/awkward),
[docs](https://awkward-array.org/)), which is an array library for arbitrary
tree-like (JSON-like) data. Because of its generality, **Awkward Array** cannot
follow the **Array API**—in fact, its array objects can't have separate `dtype`
and `shape` attributes (the array `type` can't be factorized). **Ragged** is
therefore

- a _specialization_ of **Awkward Array** for numeric data in fixed-length and
  variable-length lists, and
- a _formalization_ to adhere to the **Array API** and its fully typed
  protocols.

See
[Why does this library exist?](https://github.com/jpivarski/ragged/discussions/6)
under the [Discussions](https://github.com/jpivarski/ragged/discussions) tab for
more details.

**Ragged** is a thin wrapper around **Awkward Array**, restricting it to ragged
arrays and transforming its function arguments and return values to fit the
specification.

**Awkward Array**, in turn, is time- and memory-efficient, ready for big
datasets. Consider the following:

```python
import gc      # control for garbage collection
import psutil  # measure process memory
import time    # measure time

import math
import ragged

this_process = psutil.Process()

def measure_memory(task):
    gc.collect()
    start_memory = this_process.memory_full_info().uss
    out = task()
    gc.collect()
    stop_memory = this_process.memory_full_info().uss
    print(f"memory: {(stop_memory - start_memory) * 1e-9:.3f} GB")
    return out

def measure_time(task):
    gc.disable()
    start_time = time.perf_counter()
    out = task()
    stop_time = time.perf_counter()
    gc.enable()
    print(f"time: {stop_time - start_time:.3f} sec")
    return out

def make_big_python_object():
    out = []
    for i in range(10000000):
        out.append([j * 1.1 for j in range(i % 10)])
    return out

def make_ragged_array():
    return ragged.array(pyobj)

def compute_on_python_object():
    out = []
    for row in pyobj:
        out.append([math.sqrt(x) for x in row])
    return out

def compute_on_ragged_array():
    return ragged.sqrt(arr)
```

The `ragged.array` is 3 times smaller:

```python
>>> pyobj = measure_memory(make_big_python_object)
memory: 2.687 GB

>>> arr = measure_memory(make_ragged_array)
memory: 0.877 GB
```

and a sample calculation on it (square root of each value) is 50 times faster:

```python
>>> result = measure_time(compute_on_python_object)
time: 4.180 sec

>>> result = measure_time(compute_on_ragged_array)
time: 0.082 sec
```

**Awkward Array** and **Ragged** are generally smaller and faster than their
Python equivalents for the same reasons that **NumPy** is smaller and faster
than Python lists. See **Awkward Array**
[papers and presentations](https://awkward-array.org/doc/main/getting-started/papers-and-talks.html)
for more.

## Installation

**Ragged** is on PyPI:

```bash
pip install ragged
```

and will someday be on conda-forge.

`ragged` is a pure-Python library that only depends on `awkward` (which, in
turn, only depends on `numpy` and a compiled extension). In principle (i.e.
eventually), `ragged` can be loaded into Pyodide and JupyterLite.

# Acknowledgements

Support for this work was provided by NSF grant
[OAC-2103945](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2103945) and the
gracious help of
[Awkward Array contributors](https://github.com/scikit-hep/awkward?tab=readme-ov-file#acknowledgements).
