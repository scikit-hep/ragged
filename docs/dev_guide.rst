.. _dev_guide:

Developer Guide
===============

Repository Layout
-----------------

.. code-block:: text

    ragged/
    ├── src/ragged/
    │   ├── __init__.py                       # public API / __all__
    │   ├── _spec_array_object.py             # array class + helpers
    │   ├── _spec_creation_functions.py       # zeros, ones, arange, …
    │   ├── _spec_elementwise_functions.py    # sqrt, add, sin, …
    │   ├── _spec_manipulation_functions.py   # reshape, roll, stack, …
    │   ├── _spec_linear_algebra_functions.py # matmul, tensordot, vecdot
    │   ├── _spec_statistical_functions.py    # sum, mean, std, …
    │   ├── _spec_searching_functions.py      # argmax, nonzero, where, …
    │   ├── _spec_sorting_functions.py        # sort, argsort
    │   ├── _spec_set_functions.py            # unique_*
    │   ├── _spec_indexing_functions.py       # take
    │   ├── _spec_data_type_functions.py      # astype, can_cast, …
    │   ├── _spec_constants.py                # e, pi, inf, nan, newaxis
    │   ├── _typing.py                        # type aliases (Shape, Dtype, …)
    │   ├── _import.py                        # lazy cupy import helper
    │   ├── _helper_functions.py              # shared internal utilities
    │   └── io/
    │       ├── __init__.py
    │       └── cf.py                         # CF Conventions I/O
    ├── docs/
    │   ├── conf.py                           # Sphinx config (MyST + autodoc)
    │   ├── index.md
    │   ├── user_guide.rst
    │   └── dev_guide.rst
    ├── tests/
    │   ├── conftest.py
    │   ├── test_spec_*.py                    # spec-driven test suites
    │   └── test_*.py                         # feature-specific suites
    ├── pyproject.toml
    └── noxfile.py

Each ``_spec_*.py`` module corresponds to one section of the
`Array API specification <https://data-apis.org/array-api/latest/>`_.  Module
names mirror the spec URL slugs intentionally so grep-based cross-referencing
is easy.


The ``array`` Class
--------------------

The ``array`` class (lower-case, matching the Array API convention) lives in
``_spec_array_object.py``.

Instance attributes
~~~~~~~~~~~~~~~~~~~

Every ``array`` instance carries exactly four private attributes:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Attribute
     - Type
     - Description
   * - ``_impl``
     - ``ak.Array | numpy.ndarray | cupy.ndarray``
     - The underlying data buffer.  Almost always ``ak.Array``; scalar
       (0-D) arrays may hold a raw ``numpy.ndarray``.
   * - ``_shape``
     - ``tuple[int | None, ...]``
     - Cached shape.  Computed once by ``_shape_dtype`` and kept in sync
       manually after any mutation.
   * - ``_dtype``
     - ``numpy.dtype``
     - Cached dtype.  Derived from the leaf ``NumpyArray`` inside the layout.
   * - ``_device``
     - ``"cpu" | "cuda"``
     - String identifier for the compute backend.

These are **not** part of the public API.  Read them only inside ``_spec_*.py``
modules; external code should use the ``.shape``, ``.dtype``, ``.device``
properties.

``_new`` class method
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @classmethod
    def _new(cls, impl, shape, dtype, device) -> array:

A fast constructor that bypasses ``__init__`` validation.  Used in hot paths
such as ``__iter__`` where shape and dtype are already known.  Do not call it
from user-facing code.


Layout Types and ``_shape_dtype``
----------------------------------

Internally, ``_impl`` is an ``ak.Array`` whose **layout** is one of a small
set of Awkward Array content types:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Layout class
     - Meaning for ``ragged.array``
   * - ``NumpyArray``
     - 1-D (or packed N-D) contiguous numeric data.  ``shape`` has no
       ``None`` entries.  ``ak.to_numpy`` always succeeds.
   * - ``RegularArray``
     - Fixed inner dimension.  Produced by ``ak.from_numpy`` on an N-D
       array.  ``shape`` has no ``None`` entries.
   * - ``ListOffsetArray``
     - Variable-length rows (the common case for user-constructed arrays).
       May be truly ragged (different row lengths → ``shape[i] == None``)
       *or* incidentally uniform (all rows same length but still
       ``ListOffsetArray``).  ``ak.to_numpy`` succeeds iff all rows are
       the same length.

``_shape_dtype(layout)`` walks the layout tree once to extract ``shape`` and
``dtype``:

.. code-block:: python

    # simplified pseudocode
    def _shape_dtype(layout):
        shape = (len(layout),)
        node = layout
        while isinstance(node, ListOffsetArray | RegularArray | ListArray):
            shape += (node.size if RegularArray else None,)
            node = node.content
        # node is now NumpyArray
        return shape + node.data.shape[1:], node.data.dtype

**Key rule**: call ``_shape_dtype`` only when the layout actually changes.
After ``ak.values_astype`` (dtype cast only), shape is unchanged — update
``_dtype`` directly without re-traversing the layout.


The Box / Unbox Pattern
------------------------

Every function that consumes or produces ``ragged.array`` objects uses two
module-level helpers to move between the public type and its ``ak.Array``
implementation:

``_unbox(*inputs)``
    Extract ``._impl`` from each input ``array``.  Raises ``TypeError`` on
    mixed array subclasses or device mismatches.

    .. code-block:: python

        (impl,) = _unbox(x)
        left_impl, right_impl = _unbox(a, b)

``_box(cls, output, *, dtype=None, device=None)``
    Wrap an ``ak.Array`` result back into a ``ragged.array`` (or subclass).
    Calls ``_shape_dtype`` to populate ``_shape`` and ``_dtype``.

    .. code-block:: python

        return _box(type(x), some_ak_array)

Always use ``type(x)`` (not ``array``) as the first argument to ``_box`` so
that subclasses round-trip correctly.


Writing a New Function
-----------------------

The following checklist applies to any new Array API function or extension.

1. **Choose the right module** — pick the ``_spec_*.py`` file whose name
   matches the spec section the function belongs to.

2. **Signature** — match the Array API signature exactly (keyword-only
   arguments, ``/`` positional-only markers):

   .. code-block:: python

       def my_func(x: array, /, *, axis: int | None = None) -> array:

3. **Unbox inputs**:

   .. code-block:: python

       (impl,) = _unbox(x)

4. **Fast path for uniform arrays** — wrap the numpy equivalent in
   ``contextlib.suppress(TypeError, ValueError)`` and try ``ak.to_numpy``:

   .. code-block:: python

       with contextlib.suppress(TypeError, ValueError):
           np_arr = ak.to_numpy(impl)
           return _box(type(x), ak.from_numpy(np.my_func(np_arr)))

   ``ak.to_numpy`` succeeds for ``NumpyArray``, ``RegularArray``, and
   ``ListOffsetArray`` with uniform row lengths.  It raises ``TypeError`` or
   ``ValueError`` for genuinely ragged arrays.

5. **Ragged / general path** — implement using Awkward Array primitives
   (``ak.flatten``, ``ak.unflatten``, ``ak.num``, etc.) where possible.
   Use ``tolist()`` / list-based fallback only as a last resort for complex
   shapes that have no efficient awkward equivalent.

6. **Box the result**:

   .. code-block:: python

       return _box(type(x), result_ak)

7. **Export** — add the function name to ``__init__.py``'s ``__all__`` list
   and the relevant import block.

8. **Docstring** — include the Array API URL:

   .. code-block:: python

       """
       Short description.

       https://data-apis.org/array-api/latest/API_specification/generated/array_api.my_func.html
       """

9. **Tests** — add a ``tests/test_<feature>.py`` file (see
   :ref:`testing_conventions`).


.. _testing_conventions:

Testing Conventions
--------------------

Structure
~~~~~~~~~

Tests are grouped by feature in ``tests/test_<feature>.py``.  Within each
file, group related cases into classes:

.. code-block:: python

    class TestMyFunc1D:
        def test_basic(self): ...
        def test_dtype_preserved(self): ...

    class TestMyFunc2DRagged:
        def test_integer_index(self): ...

Helper
~~~~~~

Every test file should define a local factory to avoid repeating
``ragged.array(...)``:

.. code-block:: python

    def _make(data, dtype=None) -> ragged.array:
        return ragged.array(data, dtype=dtype)

Coverage checklist
~~~~~~~~~~~~~~~~~~

For each new function, cover:

- 1-D uniform input
- 2-D uniform input (created from ``np.ndarray``)
- 2-D ragged input (created from Python lists)
- dtype preservation (``np.float32`` should stay ``np.float32``)
- result type (``isinstance(result, ragged.array)``)
- error cases (wrong shape, wrong dtype, unsupported key type, …)
- copy / isolation (mutations via ``__setitem__`` or ``.at`` do not affect
  the original)

Running tests
~~~~~~~~~~~~~

.. code-block:: bash

    pip install -e ".[test]"
    pytest tests/

With coverage::

    pytest tests/ --cov=ragged --cov-report=term-missing

The full test matrix (multiple Python / NumPy versions) is run via ``nox``::

    nox


Performance Patterns
---------------------

The following patterns are used consistently throughout the codebase.
New code should follow them.

Single try/except for fast-path detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determine whether an array is uniform by probing ``ak.to_numpy`` once, in a
single ``try/except``.  Do not wrap value unwrapping in a separate
``try/except``; instead, branch on the result of the single probe:

.. code-block:: python

    try:
        arr_np = ak.to_numpy(self._impl)
    except (TypeError, ValueError):
        arr_np = None

    if arr_np is not None:
        # fast path — unwrap value as numpy
        val = ak.to_numpy(value._impl) if isinstance(value, array) else value
        ...
    else:
        # slow path — unwrap value as list
        val = value._impl.tolist() if isinstance(value, array) else value
        ...

Do not use ``isinstance(layout, NumpyArray | RegularArray)`` as the sole
fast-path gate — it misses ``ListOffsetArray`` arrays with incidentally
uniform rows (common when the user constructs from Python lists).

Avoid full ``tolist()`` in ragged paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prefer iterating over ``ak.Array`` sub-blocks and calling ``ak.to_numpy``
per sub-block over calling ``ak.to_list()`` on the whole array.
``ak.to_list`` allocates a Python object for every scalar; sub-block
``ak.to_numpy`` stays in C for uniform chunks.

.. code-block:: python

    # Preferred
    def _process(a: ak.Array, b: ak.Array) -> Any:
        try:
            return np_func(ak.to_numpy(a), ak.to_numpy(b))
        except (TypeError, ValueError):
            pass
        return [_process(
            ai if isinstance(ai, ak.Array) else ak.Array(ai),
            bi if isinstance(bi, ak.Array) else ak.Array(bi),
        ) for ai, bi in zip(a, b, strict=False)]

O(D) layout walks for nested structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When restoring a nested structure from a flat array (e.g. after
``ak.flatten(axis=None)``), collect counts at each nesting level with a
**single top-down walk** rather than calling ``ak.num(impl, axis=depth)``
from the root for each depth:

.. code-block:: python

    # O(D) — peel one level at a time
    level_counts: list[np.ndarray] = []
    cur = impl
    for _ in range(ndim - 1):
        level_counts.append(ak.to_numpy(ak.num(cur, axis=1)))
        cur = ak.flatten(cur, axis=1)

    result = flat_rolled
    for counts in reversed(level_counts):
        result = ak.unflatten(result, counts)

Shape is invariant under ``ak.values_astype``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a dtype cast via ``ak.values_astype``, shape does not change.  Update
``_dtype`` directly instead of re-running ``_shape_dtype``:

.. code-block:: python

    self._impl = ak.values_astype(self._impl, new_dtype)
    self._dtype = new_dtype   # shape unchanged — no _shape_dtype call needed

Zero-copy dummies for broadcast helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a dummy array is needed only to drive ``ak.broadcast_arrays`` (its
values are discarded), use a zero-copy broadcast view instead of allocating
a full array:

.. code-block:: python

    dummy = ak.from_numpy(np.broadcast_to(np.zeros((), dtype=np.int8), target_shape))


``_apply_inplace`` and in-place operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_apply_inplace`` copies ``_impl``, ``_shape``, ``_dtype``, and ``_device``
directly from the already-computed result — it does not call ``_shape_dtype``
again.  This is safe because all in-place operators (``__iadd__``, etc.) are
elementwise and therefore shape-preserving.


Awkward Array Gotchas
----------------------

``ak.from_numpy`` on N-D arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ak.from_numpy`` on a 2-D (or higher) NumPy array produces a
``NumpyArray`` layout, **not** a ``ListOffsetArray``.  The resulting
``ragged.array`` will have a concrete integer for the inner dimension
(e.g. ``shape == (3, 4)``).

However, the helper ``_ak_from_numpy`` (defined in
``_spec_manipulation_functions.py``) calls ``ak.from_regular(..., axis=None)``
afterwards to convert every regular dimension to variable-length.  Use it
when the ragged convention (``shape[-1] == None``) is required:

.. code-block:: python

    from ._spec_manipulation_functions import _ak_from_numpy
    impl = _ak_from_numpy(np_result)

``ak.flatten(axis=1)`` is O(1) for ``ListOffsetArray``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Peeling one nesting level with ``ak.flatten(impl, axis=1)`` returns the
content buffer of the outer ``ListOffsetArray`` — it does not copy data.
This makes iterative level-peeling (as in the ``roll`` axis=None path)
effectively O(1) per level.

``ak.to_numpy`` on uniform ``ListOffsetArray``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ak.to_numpy`` succeeds on a ``ListOffsetArray`` whose rows all have the
same length — it is not restricted to ``NumpyArray`` or ``RegularArray``
layouts.  This is why the fast-path probe uses ``try/except`` rather than a
layout ``isinstance`` check.
