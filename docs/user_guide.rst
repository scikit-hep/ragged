.. _user_guide:

User Guide
==========

Installation
------------

**Ragged** requires Python 3.10 or later and is available on PyPI::

    pip install ragged

The only runtime dependencies are ``awkward >= 2.6.7`` and ``numpy >= 1.24``.
GPU support (CUDA) requires ``cupy``; see `Device`_.


Quickstart
----------

.. code-block:: python

    import ragged
    import numpy as np

    # Create a ragged array from a nested Python list
    a = ragged.array([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]])

    print(a.shape)   # (3, None)
    print(a.dtype)   # float64

    # Elementwise operations preserve shape
    b = ragged.sqrt(a)

    # Boolean indexing
    c = a[a > 2.0]

    # Reduction along an axis
    row_sums = ragged.sum(a, axis=1)   # shape (3,)

    # Functional update (JAX-style, copy semantics)
    d = a.at[0].set(ragged.array([10.0, 20.0, 30.0]))


The Array Model
---------------

Shape
~~~~~

A ``ragged.array`` has a ``shape`` tuple where each element is either an
``int`` (fixed-size dimension) or ``None`` (variable-size / ragged dimension).

- ``shape[0]`` is always an ``int`` — the number of rows at the outermost level.
- Any later dimension can be ``None`` when row lengths are non-uniform.

.. code-block:: python

    ragged.array([1.0, 2.0, 3.0]).shape          # (3,)
    ragged.array([[1.0, 2.0], [3.0, 4.0]]).shape  # (2, None)  ← uniform rows
    ragged.array([[1.0, 2.0], [3.0]]).shape        # (2, None)  ← ragged rows

.. note::

    Even when all rows happen to have the same length, dimensions past the first
    report ``None`` if the array was built from Python lists.  Pass a NumPy
    array to the constructor to obtain a fully regular shape::

        ragged.array(np.ones((3, 4))).shape   # (3, 4)  ← regular from numpy

Dtype
~~~~~

``ragged.array`` always stores a single ``numpy.dtype`` for all elements.
Mixed-type input is upcasted following NumPy's type promotion rules.

.. code-block:: python

    a = ragged.array([[1, 2], [3]], dtype=np.float32)
    a.dtype   # dtype('float32')

Supported dtypes mirror the `Array API dtype table
<https://data-apis.org/array-api/latest/API_specification/data_types.html>`_:
bool, int8/16/32/64, uint8/16/32/64, float32/64, complex64/128.

Device
~~~~~~

``ragged.array`` tracks whether data live on CPU or GPU:

.. code-block:: python

    a = ragged.array([1.0, 2.0], device="cpu")   # backed by NumPy (default)
    # a = ragged.array([1.0, 2.0], device="cuda")  # backed by CuPy (needs GPU)


Creating Arrays
---------------

From Python objects
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ragged.array([[1, 2, 3], [4, 5]])       # from nested list
    ragged.array(np.arange(6).reshape(2,3)) # from numpy array
    ragged.asarray([1.0, 2.0, 3.0])         # alias for array()

From factory functions
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ragged.zeros((3,))           # 1-D zeros
    ragged.ones((2, 4))          # 2-D ones
    ragged.full((3,), 7.0)       # filled with scalar
    ragged.arange(10)            # 0..9
    ragged.linspace(0, 1, 5)     # five evenly-spaced values
    ragged.eye(3)                # 3×3 identity
    ragged.empty_like(a)         # same shape/dtype, uninitialised
    ragged.zeros_like(a)
    ragged.ones_like(a)

Meshgrid
~~~~~~~~

.. code-block:: python

    xs, ys = ragged.meshgrid(ragged.arange(3), ragged.arange(4))


Indexing
--------

Integer and slice indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    a = ragged.array([[1.0, 2.0, 3.0], [4.0, 5.0]])

    a[0]        # first row  → ragged.array([1., 2., 3.])
    a[-1]       # last row   → ragged.array([4., 5.])
    a[0:2]      # slice      → ragged.array([[1., 2., 3.], [4., 5.]])
    a[0, 1]     # element    → ragged.array(2.)   (for uniform rows)

Boolean masking
~~~~~~~~~~~~~~~

.. code-block:: python

    mask = ragged.array([True, False])
    a[mask]    # → ragged.array([[1., 2., 3.]])

    # Computed mask
    a[a > 2.0]


In-place mutation (``__setitem__``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ragged.array`` supports in-place assignment, enabling compatibility with
``array_api_extra.at``:

.. code-block:: python

    a = ragged.array([1.0, 2.0, 3.0, 4.0])
    a[1:3] = 0.0                             # slice ← scalar
    a[ragged.array([True, False, True, False])] = 99.0  # boolean mask

For ragged arrays, in-place assignment supports integer and slice keys on the
outermost axis (boolean-mask keys raise ``TypeError`` on ragged layouts):

.. code-block:: python

    r = ragged.array([[1.0, 2.0], [3.0]])
    r[0] = [10.0, 20.0, 30.0]    # replace row with a different-length row


Functional updates with ``.at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``.at`` interface provides JAX-style copy-semantics updates — the original
array is never mutated:

.. code-block:: python

    a = ragged.array([1.0, 2.0, 3.0])

    a.at[1].set(99.0)            # → [1., 99., 3.]
    a.at[0:2].add(10.0)          # → [11., 12., 3.]
    a.at[2].multiply(2.0)        # → [1., 2., 6.]
    a.at[1].subtract(0.5)
    a.at[1].divide(2.0)
    a.at[2].power(2.0)
    a.at[1].min(3.0)             # x[1] = min(x[1], 3.0)
    a.at[1].max(0.0)             # x[1] = max(x[1], 0.0)

    print(a)   # [1., 2., 3.]  ← unchanged


Elementwise Operations
-----------------------

All `Array API elementwise functions
<https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html>`_
are available:

.. code-block:: python

    ragged.sqrt(a)
    ragged.abs(a)
    ragged.exp(a)
    ragged.log(a)
    ragged.sin(a); ragged.cos(a); ragged.tan(a)
    ragged.add(a, b)          # or a + b
    ragged.multiply(a, 2.0)   # or a * 2.0
    ragged.equal(a, b)        # or a == b

Operator overloads
~~~~~~~~~~~~~~~~~~

All standard Python operators are supported:
``+``, ``-``, ``*``, ``/``, ``//``, ``**``, ``%``,
``&``, ``|``, ``^``, ``~``, ``<<``, ``>>``,
``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``,
``@`` (matrix multiply).

NumPy interoperability (NEP-13 / NEP-18)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ragged.array`` implements ``__array_ufunc__`` (NEP-13) and
``__array_function__`` (NEP-18), so NumPy functions work transparently:

.. code-block:: python

    import numpy as np

    np.sqrt(a)                   # delegates to ragged.sqrt
    np.add(a, b)                 # delegates through __array_ufunc__
    np.concatenate([a, b])       # delegates to ragged.concat
    np.stack([a, b])             # delegates to ragged.stack


Array Manipulation
------------------

Shape operations
~~~~~~~~~~~~~~~~

.. code-block:: python

    ragged.reshape(a, (6,))         # reshape (uniform arrays only)
    ragged.squeeze(a, axis=0)       # remove size-1 axis
    ragged.expand_dims(a, axis=0)   # insert new axis
    ragged.permute_dims(a, (1, 0))  # transpose / axis permutation

Reordering
~~~~~~~~~~

.. code-block:: python

    ragged.flip(a)                  # reverse all elements
    ragged.flip(a, axis=0)          # reverse along axis 0
    ragged.roll(a, shift=2)         # roll all elements by 2
    ragged.roll(a, shift=1, axis=0) # roll rows

Joining
~~~~~~~

.. code-block:: python

    ragged.concat([a, b])           # concatenate along axis 0
    ragged.concat([a, b], axis=1)   # concatenate along axis 1
    ragged.stack([a, b])            # stack into new axis 0
    ragged.stack([a, b], axis=1)

Broadcasting
~~~~~~~~~~~~

.. code-block:: python

    ragged.broadcast_to(a, (4, 3))
    x, y = ragged.broadcast_arrays(a, b)


Statistical and Searching Functions
------------------------------------

.. code-block:: python

    ragged.sum(a)                  # sum of all elements
    ragged.sum(a, axis=1)          # sum per row
    ragged.mean(a, axis=0)
    ragged.max(a); ragged.min(a)
    ragged.std(a); ragged.var(a)
    ragged.prod(a)

    ragged.argmax(a); ragged.argmin(a)
    ragged.nonzero(a)
    ragged.where(condition, x, y)

    ragged.sort(a)
    ragged.argsort(a)


Linear Algebra
--------------

.. code-block:: python

    ragged.matmul(a, b)            # or a @ b
    ragged.tensordot(a, b, axes=1)
    ragged.vecdot(a, b)            # inner product over last axis
    ragged.matrix_transpose(a)     # swap last two axes


Data Type Functions
-------------------

.. code-block:: python

    ragged.astype(a, np.float32)
    ragged.can_cast(np.float32, np.float64)
    ragged.result_type(a, b)
    ragged.isdtype(np.float32, "real floating")


I/O: CF Conventions
--------------------

The ``ragged.io`` sub-package provides serialisation helpers for the
`Climate and Forecast (CF) Conventions
<https://cfconventions.org/>`_ ragged-array encodings.

Contiguous ragged array (CF DSG H.3.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A *contiguous* encoding stores all values in a flat ``content`` array and
row lengths in a ``counts`` array:

.. code-block:: python

    import ragged.io

    a = ragged.array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]])

    content, counts = ragged.io.to_cf_contiguous(a)
    # content: ragged.array([1., 2., 3., 4., 5., 6.])
    # counts:  ragged.array([2, 1, 3])

    restored = ragged.io.from_cf_contiguous(content, counts)

Indexed ragged array (CF DSG H.4.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An *indexed* encoding stores the row index for every element:

.. code-block:: python

    content, index = ragged.io.to_cf_indexed(a)
    # content: ragged.array([1., 2., 3., 4., 5., 6.])
    # index:   ragged.array([0, 0, 1, 2, 2, 2])

    restored = ragged.io.from_cf_indexed(content, index)


Constants
---------

.. code-block:: python

    ragged.e          # Euler's number
    ragged.pi         # π
    ragged.inf        # IEEE 754 infinity
    ragged.nan        # IEEE 754 NaN
    ragged.newaxis    # alias for None, for use in indexing


Array API Compliance
--------------------

``ragged`` targets the `Python Array API Standard
<https://data-apis.org/array-api/latest/>`_ (2022.12 and later).
The namespace is discoverable via:

.. code-block:: python

    xp = a.__array_namespace__()   # returns the ragged module
    xp.sqrt(a)

This allows Array-API-consuming libraries (e.g. ``array_api_extra``,
``scipy``, ``sklearn``) to use ``ragged.array`` transparently wherever they
accept an Array API input.
