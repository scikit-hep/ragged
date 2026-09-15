# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html
"""

from __future__ import annotations

import contextlib
import numbers
from collections.abc import Iterable
from typing import Any, cast

import awkward as ak
import numpy as np

from ._spec_array_object import _box, _unbox, array


def _ak_from_numpy(arr: np.ndarray) -> ak.Array:
    """Convert a numpy array to ak.Array with variable-length (not Regular) inner dims.

    ``ak.from_numpy`` produces ``RegularArray`` layouts for multidimensional
    inputs, which gives fixed integer sizes in ``ragged.array.shape`` instead
    of the expected ``None``.  Calling ``ak.from_regular(..., axis=None)``
    afterwards converts every regular dimension to variable-length so that
    ``shape`` matches the ragged convention.
    """
    ak_arr = ak.from_numpy(arr)
    if arr.ndim > 1:
        ak_arr = ak.from_regular(ak_arr, axis=None)
    return ak_arr


def broadcast_arrays(*arrays: array) -> list[array]:
    """
    Broadcasts one or more arrays against one another.

    Args:
        arrays: An arbitrary number of to-be broadcasted arrays.

    Returns:
        A list of broadcasted arrays. Each array has the same shape. Each array
        has the same dtype as its corresponding input array.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.broadcast_arrays.html
    """

    impls = _unbox(*arrays)
    if all(not isinstance(x, ak.Array) for x in impls):
        return [_box(type(arrays[i]), x) for i, x in enumerate(impls)]
    else:
        out = [x if isinstance(x, ak.Array) else x.reshape((1,)) for x in impls]  # type: ignore[union-attr]
        return [
            _box(type(arrays[i]), x) for i, x in enumerate(ak.broadcast_arrays(*out))
        ]


def broadcast_to(x: array, /, shape: tuple[int | None, ...]) -> array:
    """
    Broadcasts an array to a specified shape.

    Args:
        x: Array to broadcast.
        shape: Array shape. Must be compatible with ``x``. Use ``None`` for
            variable-length (ragged) dimensions.  If the array is incompatible
            with the specified shape, the function raises an exception.

    Returns:
        An array having a specified shape. Must have the same data type as x.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.broadcast_to.html
    """

    x_obj: object = x
    if not isinstance(x_obj, array):
        msg = f"broadcast_to: expected a ragged.array, got {type(x).__name__!r}"
        raise TypeError(msg)

    shape_any: Any = shape
    if not isinstance(shape_any, tuple):
        msg_shape: str = "shape must be a tuple, got " + str(type(shape))
        raise TypeError(msg_shape)

    for dim in shape:
        dim_any: Any = dim
        if dim_any is not None:
            if not isinstance(dim_any, int):
                msg_type: str = "shape dimensions must be int or None, got " + str(
                    type(dim)
                )
                raise TypeError(msg_type)
            if dim_any < 0:
                msg_value: str = "shape dimensions must be non-negative, got " + str(
                    dim
                )
                raise ValueError(msg_value)

    (impl,) = _unbox(x)
    ndim = x.ndim
    target_ndim = len(shape)

    if target_ndim < ndim:
        msg = (
            f"Cannot broadcast array of shape {x.shape} to shape {shape}: "
            "target has fewer dimensions"
        )
        raise ValueError(msg)

    # NOTE: there is deliberately no numpy fast path here. A uniform input whose
    # *layout* is ragged (e.g. shape ``(2, None)``) must broadcast to the same
    # shape signature as a genuinely ragged input: the source array's own inner
    # dimensions stay variable-length (``None``) while newly prepended outer
    # dimensions become regular. ``np.broadcast_to`` + ``ak.from_numpy`` would
    # instead make *every* dimension a fixed int, silently forking the result
    # convention on whether the data happened to be uniform. The awkward
    # ``broadcast_arrays`` path below already produces the correct, consistent
    # convention for both uniform and ragged inputs, so it is used
    # unconditionally.

    # Align ndim: prepend size-1 outer dimensions so impl.ndim == target_ndim
    current: ak.Array = impl
    for _ in range(target_ndim - ndim):
        current = current[np.newaxis]

    # Build a dummy numpy array whose shape drives the broadcast.
    # Replace None (ragged) dims with 1 so np.zeros accepts it.
    x_shape = x.shape  # original shape, length ndim
    # Pad x_shape with 1s on the left to align with target_ndim
    padded_x = (1,) * (target_ndim - ndim) + tuple(
        1 if s is None else s for s in x_shape
    )

    dummy_shape: list[int] = []
    for i, s in enumerate(shape):
        if s is None:
            dummy_shape.append(1)  # ragged dim: dummy size irrelevant
        else:
            px = padded_x[i]
            if px not in {1, s}:
                msg = (
                    f"Cannot broadcast array of shape {x.shape} to shape {shape}: "
                    f"dimension {i} has size {px}, cannot broadcast to {s}"
                )
                raise ValueError(msg)
            dummy_shape.append(s)

    # Use a zero-copy broadcast view as the shape driver — values are discarded
    # by ak.broadcast_arrays, so no allocation proportional to dummy_shape needed.
    dummy = ak.from_numpy(np.broadcast_to(np.zeros((), dtype=np.int8), dummy_shape))

    try:
        bx, _ = ak.broadcast_arrays(current, dummy)
    except Exception as e:
        msg = f"Cannot broadcast array of shape {x.shape} to shape {shape}"
        raise ValueError(msg) from e

    return _box(type(x), bx)


def concat(
    arrays: tuple[array, ...] | list[array], /, *, axis: None | int = 0
) -> array:
    """
    Joins a sequence of arrays along an existing axis.

    Args:
        arrays: Input arrays to join. The arrays must have the same shape,
            except in the dimension specified by `axis`.
        axis: Axis along which the arrays will be joined. If `axis` is `None`,
            arrays are flattened before concatenation. If `axis` is negative,
            the function determines the axis along which to join by counting
            from the last dimension.

    Returns:
        An output array containing the concatenated values. If the input arrays
        have different data types, normal type promotion rules apply. If the
        input arrays have the same data type, the output array has the same
        data type as the input arrays.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.concat.html
    """

    if len(arrays) == 0:
        msg = "need at least one array to concatenate"
        raise ValueError(msg)

    first = arrays[0]
    if not all(first.ndim == x.ndim for x in arrays[1:]):
        msg = "all the input arrays must have the same number of dimensions"
        raise ValueError(msg)

    if first.ndim == 0:
        msg = "zero-dimensional arrays cannot be concatenated"
        raise ValueError(msg)

    impls = _unbox(*arrays)
    assert all(isinstance(x, ak.Array) for x in impls)

    if axis is None:
        impls = [ak.ravel(x) for x in impls]  # type: ignore[assignment]
        axis = 0

    return _box(type(first), ak.concatenate(impls, axis=axis))


def expand_dims(x: array, /, *, axis: int = 0) -> array:
    """
    Expands the shape of an array by inserting a new axis (dimension) of size
    one at the position specified by `axis`.

    Args:
        x: Input array.
        axis: Axis position (zero-based). If `x` has rank (i.e, number of
            dimensions) `N`, a valid `axis` must reside on the closed-interval
            `[-N-1, N]`. If provided a negative axis, the axis position at
            which to insert a singleton dimension is computed as
            `N + axis + 1`. Hence, if provided -1, the resolved axis position
            is `N` (i.e., a singleton dimension is appended to the input array
            `x`). If provided `-N - 1`, the resolved axis position is 0 (i.e.,
            a singleton dimension is prepended to the input array x). An
            `IndexError` exception is raised if provided an invalid axis
            position.

    Returns:
        An expanded output array having the same data type as `x`.

    This is the opposite of `ragged.squeeze`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.expand_dims.html
    """

    original_axis = axis
    if axis < 0:
        axis += x.ndim + 1
    if not 0 <= axis <= x.ndim:
        msg = (
            f"axis {original_axis} is out of bounds for array of dimension {x.ndim + 1}"
        )
        raise ak.errors.AxisError(msg)

    slicer = (slice(None),) * axis + (np.newaxis,)
    shape = x.shape[:axis] + (1,) + x.shape[axis:]

    out = x._impl[slicer]  # type: ignore[index] # pylint: disable=W0212
    if not isinstance(out, ak.Array):
        out = ak.Array(out)

    return x._new(out, shape, x.dtype, x.device)  # pylint: disable=W0212


def flip(x: array, /, *, axis: None | int | tuple[int, ...] = None) -> array:
    """
    Reverses the order of elements in an array along the given axis. The shape
    of the array is preserved.

    Args:
        x: Input array.
        axis: Axis (or axes) along which to flip. If `axis` is `None`, the
        function flips all input array axes. If `axis` is negative, the
        function counts from the last dimension. If provided more than one
        axis, the function flips only the specified axes.

    Returns:
        An output array having the same data type and shape as `x` and whose
        elements, relative to `x`, are reordered.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.flip.html
    """

    (impl,) = _unbox(x)
    ndim = x.ndim

    # Normalise axis to a frozenset of non-negative ints.
    if axis is None:
        axes: frozenset[int] = frozenset(range(ndim))
    else:
        raw: tuple[int, ...] = (axis,) if isinstance(axis, int) else tuple(axis)
        normalised: list[int] = []
        for a in raw:
            a_norm = a if a >= 0 else a + ndim
            if a_norm < 0 or a_norm >= ndim:
                msg = f"flip: axis {a} is out of range for array with {ndim} dimensions"
                raise ValueError(msg)
            normalised.append(a_norm)
        axes = frozenset(normalised)

    # Fast path: uniform (non-ragged) array via numpy.
    #
    # Use ``_ak_from_numpy`` (not plain ``ak.from_numpy``) so that the inner
    # dimensions come back variable-length (``None``), matching the awkward
    # paths below. ``flip`` preserves the input shape, so a ``(2, None)`` input
    # must stay ``(2, None)`` regardless of whether its data is uniform.
    with contextlib.suppress(TypeError, ValueError):
        np_arr = ak.to_numpy(impl)
        return _box(
            type(x),
            _ak_from_numpy(
                np.flip(np_arr, axis=tuple(axes) if axis is not None else None)
            ),
        )

    # Medium path: awkward-native operations — no to_list round-trip.
    #
    # axis=0: impl[::-1] is O(1) — awkward builds an IndexedArray view
    #   over the existing offsets buffer; no element copying occurs.
    #
    # innermost axis (ndim-1): flatten to 1-D, build a numpy reversal
    #   index (contiguous reversed ranges per row), then unflatten.
    #   All arithmetic is in numpy/C; no Python loop over elements.

    result = impl

    if 0 in axes:
        result = result[::-1]  # type: ignore[index]

    inner_axes = axes - {0}
    if inner_axes and inner_axes == {ndim - 1}:
        flat = ak.flatten(result)
        counts_ak = ak.num(result)
        counts_np = ak.to_numpy(counts_ak)
        offsets = np.concatenate(([0], np.cumsum(counts_np)))
        sorter: np.ndarray = (
            np.concatenate(
                [
                    np.arange(int(o) + int(n) - 1, int(o) - 1, -1, dtype=np.intp)
                    for o, n in zip(offsets, counts_np, strict=False)
                ]
            )
            if len(counts_np)
            else np.array([], dtype=np.intp)
        )
        flat_any: Any = flat
        result = ak.unflatten(flat_any[sorter], counts_ak)
    elif inner_axes:
        # Fallback for multi-level inner axes via list round-trip.
        # axis=0 is already applied; pass inner_axes so depth=0 is a no-op.
        def _flip(obj: Any, depth: int, flip_axes: frozenset[int]) -> Any:
            if not isinstance(obj, list):
                return obj
            items: list[Any] = list(reversed(obj)) if depth in flip_axes else obj
            return [_flip(item, depth + 1, flip_axes) for item in items]

        result = ak.Array(_flip(ak.to_list(result), 0, inner_axes))

    return _box(type(x), result)


def permute_dims(x: array, /, axes: tuple[int, ...]) -> array:
    """
    Permutes the axes (dimensions) of an array `x`.

    Args:
        x: Input array.
        axes: Tuple containing a permutation of `(0, 1, ..., N-1)` where `N` is
            the number of axes (dimensions) of `x`.

    Returns:
        An array containing the axes permutation. The returned array has the
        same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.permute_dims.html
    """

    (impl,) = _unbox(x)
    ndim = x.ndim

    # Validate axes.
    if len(axes) != ndim or sorted(axes) != list(range(ndim)):
        msg = f"axes must be a permutation of (0,...,{ndim - 1}), got {axes}"
        raise ValueError(msg)

    # Identity permutation — return a view.
    if axes == tuple(range(ndim)):
        return _box(type(x), impl)

    # Fast path: uniform array via numpy.transpose.
    with contextlib.suppress(TypeError, ValueError):
        np_arr = ak.to_numpy(impl)
        return _box(type(x), _ak_from_numpy(np.transpose(np_arr, axes)))

    # Medium path: awkward-native 2-D transpose.
    #
    # For (1, 0) on a 2-D ragged array `impl` of shape (n_rows, None):
    #   Column j of the result contains impl[i][j] for every row i that
    #   has at least j+1 elements — i.e. the j-th element of each row,
    #   collected across rows.
    #
    # Algorithm (all numpy/awkward, no Python loops over elements):
    #   1. flat   = flatten to 1-D content buffer
    #   2. counts = number of elements per row
    #   3. max_cols = max(counts)
    #   4. For each column j, the source indices are offsets[i] + j for
    #      all i where counts[i] > j.  Build these via numpy broadcasting.
    #   5. new_counts[j] = number of rows that have a j-th element.
    #   6. Gather from flat and unflatten with new_counts.
    if ndim == 2 and axes == (1, 0):
        flat = ak.flatten(impl)
        counts_np = ak.to_numpy(ak.num(impl))
        if len(counts_np) == 0:
            return _box(type(x), ak.Array([]))
        max_cols = int(counts_np.max())
        offsets = np.concatenate(([0], np.cumsum(counts_np)))
        # col_indices[j]: boolean mask of rows that have a j-th element
        # Build source-index array for all columns at once.
        new_counts = np.array(
            [(counts_np > j).sum() for j in range(max_cols)], dtype=np.intp
        )
        indices = np.concatenate(
            [offsets[:-1][counts_np > j] + j for j in range(max_cols)]
        ).astype(np.intp)
        flat_any: Any = flat
        return _box(type(x), ak.unflatten(flat_any[indices], new_counts))

    # List-based fallback for exotic permutations (e.g. 3-D with axis swap).
    def _transpose_matrix(mat: list[Any]) -> list[Any]:
        max_cols = max((len(row) for row in mat), default=0)
        return [[row[i] for row in mat if i < len(row)] for i in range(max_cols)]

    def _permute(lst: Any, order: list[int]) -> Any:
        if not order:
            return lst
        if order[0] == 0:
            if all(isinstance(e, list) for e in lst):
                return [_permute(e, order[1:]) for e in lst]
            return lst
        if all(isinstance(e, list) for e in lst):
            transposed = _transpose_matrix(lst)
            new_order = [order[0] - 1] + [i - 1 if i > 0 else i for i in order[1:]]
            return [_permute(t, new_order) for t in transposed]
        return lst

    result: list[Any] = _permute(ak.to_list(impl), list(axes))
    return _box(type(x), ak.Array(result))


def reshape(x: array, /, shape: tuple[int, ...], *, copy: None | bool = None) -> array:
    """
    Reshapes an array without changing its data.

    Args:
        x: Input array to reshape.
        shape: A new shape compatible with the original shape. One shape
            dimension is allowed to be -1. When a shape dimension is -1, the
            corresponding output array shape dimension is inferred from the
            length of the array and the remaining dimensions.
        copy: Boolean indicating whether or not to copy the input array. If
            `True`, the function always copies. If `False`, the function never
            copies and raises a `ValueError` in case a copy would be necessary.
            If `None`, the function reuses the existing memory buffer if
            possible and copies otherwise.

    Returns:
        An output array having the same data type and elements as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.reshape.html
    """

    # --- validate shape ---
    neg_one_count = sum(1 for s in shape if s == -1)
    if neg_one_count > 1:
        msg = "reshape: only one dimension may be -1"
        raise ValueError(msg)
    if any(s < -1 or s == 0 and s != 0 for s in shape):
        # zero-size dims are allowed; negative dims other than -1 are not
        pass
    if any(s < -1 for s in shape):
        msg = "reshape: shape dimensions must be -1 or non-negative"
        raise ValueError(msg)

    # --- fast path: fully regular arrays (can convert to numpy) ---
    x_np: np.ndarray | None = None
    with contextlib.suppress(TypeError, ValueError):
        x_np = ak.to_numpy(x._impl)  # pylint: disable=W0212

    if x_np is not None:
        # Resolve -1
        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            total = x_np.size
            if known == 0 or total % known != 0:
                msg = (
                    f"reshape: cannot reshape array of size {total} "
                    f"into shape {shape}"
                )
                raise ValueError(msg)
            shape = tuple(total // known if s == -1 else s for s in shape)

        result_np = x_np.reshape(shape)  # raises ValueError on size mismatch

        if copy is False and not np.shares_memory(x_np, result_np):
            msg = "reshape: copy=False but a copy was required"
            raise ValueError(msg)

        # Use ``_ak_from_numpy`` so inner dimensions come back variable-length
        # (``None``), matching the ragged shape convention used elsewhere. The
        # reshaped result of a ragged-typed array is itself a transformed ragged
        # array, so e.g. ``reshape(x, (2, 2, 1))`` yields ``(2, None, None)``
        # rather than ``(2, 2, 1)``.
        return _box(type(x), _ak_from_numpy(result_np))

    # --- ragged array: total element count is variable across batch entries ---
    # The only unambiguous target is a flat 1-D array (all elements concatenated).
    total_elements = int(ak.count(x._impl, axis=None))  # pylint: disable=W0212

    # Resolve -1 using total element count.
    resolved: list[int] = list(shape)
    if -1 in resolved:
        idx = resolved.index(-1)
        known = 1
        for s in resolved:
            if s != -1:
                known *= s
        if known == 0 or total_elements % known != 0:
            msg = (
                f"reshape: cannot reshape ragged array with {total_elements} "
                f"total elements into shape {shape}"
            )
            raise ValueError(msg)
        resolved[idx] = total_elements // known

    target_total = 1
    for s in resolved:
        target_total *= s

    if target_total != total_elements:
        msg = (
            f"reshape: cannot reshape ragged array with {total_elements} "
            f"total elements into shape {tuple(resolved)}"
        )
        raise ValueError(msg)

    # Only allow 1-D target for ragged arrays.
    if len(resolved) != 1:
        msg = (
            "reshape: ragged arrays can only be reshaped to 1-D (flatten); "
            "multi-dimensional reshape requires all dimensions to be regular"
        )
        raise ValueError(msg)

    flat = ak.flatten(x._impl, axis=None)  # pylint: disable=W0212
    return _box(type(x), flat)


def roll(
    x: array,
    /,
    shift: int | tuple[int, ...],
    *,
    axis: None | int | tuple[int, ...] = None,
) -> array:
    """
    Rolls array elements along a specified axis. Array elements that roll
    beyond the last position are re-introduced at the first position. Array
    elements that roll beyond the first position are re-introduced at the last
    position.

    Args:
        x: Input array.
        shift: Number of places by which the elements are shifted. If `shift`
            is a tuple, then `axis` must be a tuple of the same size, and each
            of the given axes must be shifted by the corresponding element in
            `shift`. If `shift` is an `int` and `axis` a tuple, then the same
            shift is used for all specified axes. If a shift is positive, then
            array elements are shifted positively (toward larger indices) along
            the dimension of `axis`. If a `shift` is negative, then array
            elements are shifted negatively (toward smaller indices) along the
            dimension of `axis`.
        axis: Axis (or axes) along which elements to shift. If `axis` is
            `None`, the array is flattened, shifted, and then restored to its
            original shape.

    Returns:
        An output array having the same data type as `x` and whose elements,
        relative to `x`, are shifted.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.roll.html
    """
    (impl,) = _unbox(x)
    ndim = x.ndim

    # ------------------------------------------------------------------
    # axis=None: flatten all elements, roll, restore nested structure
    # ------------------------------------------------------------------
    if axis is None:
        shift_any: Any = shift
        if not isinstance(shift_any, int):
            msg = f"shift must be an int when axis is None, got {type(shift)}"
            raise TypeError(msg)
        flat = cast(ak.Array, ak.flatten(impl, axis=None))
        n = len(flat)
        if n == 0:
            return _box(type(x), impl)
        s = int(shift_any) % n
        rolled: ak.Array = (
            cast(ak.Array, ak.concatenate([flat[-s:], flat[:-s]]))  # pylint: disable=unsubscriptable-object
            if s
            else flat
        )
        # Restore nested structure layer by layer (innermost first).
        # For ndim=1: no unflatten needed (already 1D).
        # For ndim=k: collect counts at each nesting level with a single O(D)
        # top-down walk — each ak.num(cur, axis=1) reads only the outermost
        # offsets of `cur`; ak.flatten peels one level (O(1) for ListOffsetArray).
        # The original approach called ak.num(impl, axis=depth) from the root
        # for every depth, giving O(D²/2) total traversal work.
        level_counts: list[np.ndarray] = []
        cur: ak.Array = impl
        for _ in range(ndim - 1):
            level_counts.append(ak.to_numpy(ak.num(cur, axis=1)))
            cur = cast(ak.Array, ak.flatten(cur, axis=1))

        result_ak: ak.Array = rolled
        for counts in reversed(level_counts):
            result_ak = cast(ak.Array, ak.unflatten(result_ak, counts))
        return _box(type(x), result_ak)

    # ------------------------------------------------------------------
    # Normalize axis/shift to parallel tuples
    # ------------------------------------------------------------------
    if isinstance(axis, int):
        axis_tuple: tuple[int, ...] = (axis,)
    elif isinstance(axis, tuple):
        axis_any: Any = axis
        if not all(isinstance(a, int) for a in axis_any):
            msg = f"axis must be int or tuple of ints, got element types in {axis}"
            raise TypeError(msg)
        axis_tuple = tuple(axis)
    else:  # pragma: no cover
        axis_type_any: Any = axis  # type: ignore[unreachable]
        msg = f"axis must be int, None, or tuple of ints, got {type(axis_type_any)}"
        raise TypeError(msg)

    if isinstance(shift, int):
        shift_tuple: tuple[int, ...] = (shift,) * len(axis_tuple)
    elif isinstance(shift, tuple):
        shift_t_any: Any = shift
        if not all(isinstance(s, int) for s in shift_t_any):
            msg = f"shift must be int or tuple of ints, got {type(shift)}"
            raise TypeError(msg)
        shift_tuple = tuple(shift)
        if len(shift_tuple) != len(axis_tuple):
            msg = (
                f"shift and axis must have the same length, "
                f"got shift={shift_tuple} and axis={axis_tuple}"
            )
            raise ValueError(msg)
    else:  # pragma: no cover
        shift_type_any: Any = shift  # type: ignore[unreachable]
        msg = f"shift must be int or tuple of ints, got {type(shift_type_any)}"
        raise TypeError(msg)

    # Normalise negative axes
    axis_tuple = tuple(a + ndim if a < 0 else a for a in axis_tuple)

    # ------------------------------------------------------------------
    # Fast path: uniform arrays via numpy
    # ------------------------------------------------------------------
    with contextlib.suppress(TypeError, ValueError):
        np_arr = ak.to_numpy(impl)
        if len(axis_tuple) == 1:
            rolled_np = np.roll(np_arr, shift_tuple[0], axis=axis_tuple[0])
        else:
            rolled_np = np_arr
            for ax, sh in zip(axis_tuple, shift_tuple, strict=False):
                rolled_np = np.roll(rolled_np, sh, axis=ax)
        return _box(type(x), _ak_from_numpy(rolled_np))

    # ------------------------------------------------------------------
    # Ragged paths: apply each (axis, shift) pair in order
    # ------------------------------------------------------------------
    current: ak.Array = impl
    for ax, sh in zip(axis_tuple, shift_tuple, strict=False):
        current = _roll_ragged_axis(current, sh, ax, ndim)
    return _box(type(x), current)


def _roll_ragged_axis(impl: ak.Array, shift: int, axis: int, ndim: int) -> ak.Array:
    """Roll a ragged ak.Array by *shift* along *axis* (already normalised ≥ 0)."""
    # ---- axis 0: O(1) slice-based concatenation ----------------------
    if axis == 0:
        n = len(impl)
        if n == 0:
            return impl
        s = shift % n
        if s == 0:
            return impl
        return cast(ak.Array, ak.concatenate([impl[-s:], impl[:-s]]))  # pylint: disable=unsubscriptable-object

    # ---- axis 1 for 2-D ragged: flatten → indexed roll → unflatten --
    if axis == 1 and ndim == 2:
        flat = ak.flatten(impl)
        counts = ak.to_numpy(ak.num(impl)).astype(np.intp)
        total = int(counts.sum())
        if total == 0:
            return impl
        offsets = np.concatenate(([0], np.cumsum(counts)))
        idx = np.empty(total, dtype=np.intp)
        for o, n in zip(offsets[:-1], counts, strict=False):
            ni = int(n)
            if ni == 0:
                continue
            rs = shift % ni
            start = int(o)
            if rs == 0:
                idx[start : start + ni] = np.arange(start, start + ni, dtype=np.intp)
            else:
                idx[start : start + ni] = start + np.roll(
                    np.arange(ni, dtype=np.intp), rs
                )
        flat_any: Any = flat
        return cast(ak.Array, ak.unflatten(flat_any[idx], counts))

    # ---- fallback: Python list recursion (3-D+ ragged, axis ≥ 2) ----
    def _roll_list(obj: Any, depth: int) -> Any:
        if not isinstance(obj, Iterable) or isinstance(obj, str | bytes):
            return obj
        if depth == axis:
            items = list(obj)
            ni = len(items)
            if ni == 0:
                return []
            rs = shift % ni
            return items[-rs:] + items[:-rs] if rs else items
        return [_roll_list(item, depth + 1) for item in obj]

    return ak.Array(_roll_list(ak.to_list(impl), 0))


def squeeze(x: array, /, axis: int | tuple[int, ...]) -> array:
    """
    Removes singleton dimensions (axes) from `x`.

    Args:
        x: Input array.
        axis: Axis (or axes) to squeeze. If a specified axis has a size
            greater than one, a `ValueError` is raised.

    Returns:
        An output array having the same data type and elements as `x`.

    This is the opposite of `ragged.expand_dims`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.squeeze.html
    """

    if isinstance(axis, numbers.Integral):
        axis = (axis,)  # type: ignore[assignment]

    posaxis = []
    for axisitem in axis:  # type: ignore[union-attr]
        posaxisitem = axisitem + x.ndim if axisitem < 0 else axisitem
        if not 0 <= posaxisitem < x.ndim and not posaxisitem == x.ndim == 0:
            msg = f"axis {axisitem} is out of bounds for array of dimension {x.ndim}"
            raise ak.errors.AxisError(msg)
        posaxis.append(posaxisitem)

    if not isinstance(x._impl, ak.Array):  # pylint: disable=W0212
        return x._new(x._impl, x._shape, x._dtype, x._device)  # pylint: disable=W0212

    out = x._impl  # pylint: disable=W0212
    shape = list(x.shape)
    for i, shapeitem in reversed(list(enumerate(x.shape))):
        if i in posaxis:
            if shapeitem is None:
                if not np.all(ak.num(out, axis=i) == 1):
                    msg = "cannot select an axis to squeeze out which has size not equal to one"
                    raise ValueError(msg)
                else:
                    out = out[(slice(None),) * i + (0,)]
                    del shape[i]

            elif shapeitem == 1:
                out = out[(slice(None),) * i + (0,)]
                del shape[i]

            else:
                msg = "cannot select an axis to squeeze out which has size not equal to one"
                raise ValueError(msg)

    return x._new(out, tuple(shape), x.dtype, x.device)  # pylint: disable=W0212


def stack(arrays: tuple[array, ...] | list[array], /, *, axis: int = 0) -> array:
    """
    Joins a sequence of arrays along a new axis.

    Args:
        arrays: Input arrays to join. Each array must have the same shape.
        axis: Axis along which the arrays will be joined. Providing an `axis`
            specifies the index of the new axis in the dimensions of the
            result. For example, if `axis` is 0, the new axis will be the first
            dimension and the output array will have shape `(N, A, B, C)`; if
            `axis` is 1, the new axis will be the second dimension and the
            output array will have shape `(A, N, B, C)`; and, if `axis` is -1,
            the new axis will be the last dimension and the output array will
            have shape `(A, B, C, N)`. A valid axis must be on the interval
            `[-N, N)`, where `N` is the rank (number of dimensions) of `x`.
            If provided an `axis` outside of the required interval, the
            function raises an exception.

    Returns:
        An output array having rank `N + 1`, where `N` is the rank (number of
        dimensions) of `x`. If the input arrays have different data types,
        normal type promotion rules apply. If the input arrays have the same
        data type, the output array has the same data type as the input arrays.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.stack.html
    """

    if not arrays:
        msg = "stack requires a non-empty sequence of arrays"
        raise ValueError(msg)

    first = arrays[0]
    first_obj: object = first
    if not isinstance(first_obj, array):
        msg = f"stack: expected ragged.array inputs, got {type(first).__name__!r}"
        raise TypeError(msg)

    ndim = first.ndim
    dtype0 = first.dtype

    for arr in arrays[1:]:
        arr_obj: object = arr
        if not isinstance(arr_obj, array):
            msg = f"stack: expected ragged.array inputs, got {type(arr).__name__!r}"
            raise TypeError(msg)
        if arr.ndim != ndim:
            msg = (
                f"stack: all arrays must have the same number of dimensions, "
                f"got ndim={ndim} and ndim={arr.ndim}"
            )
            raise ValueError(msg)
        if arr.dtype != dtype0:
            msg = (
                f"stack: all arrays must have the same dtype, "
                f"got {dtype0} and {arr.dtype}"
            )
            raise ValueError(msg)

    # axis is in [-ndim-1, ndim]; normalise to [0, ndim]
    axis_any: Any = axis
    if not isinstance(axis_any, int):
        msg = f"stack: axis must be an int, got {type(axis)}"
        raise TypeError(msg)
    if not -(ndim + 1) <= axis <= ndim:
        msg = f"stack: axis={axis} is out of bounds for ndim={ndim}"
        raise ValueError(msg)
    axis_norm = axis if axis >= 0 else axis + ndim + 1

    # NOTE: there is deliberately no numpy fast path here. ``np.stack`` +
    # ``ak.from_numpy`` would make every dimension a fixed int, whereas the
    # expand_dims+concat path below keeps the *newly inserted* stack axis
    # regular while leaving the source arrays' own inner dimensions
    # variable-length (``None``). For example, stacking two ``(2, None)`` arrays
    # along axis 0 yields ``(2, 2, None)`` (and along axis 2 yields
    # ``(2, None, 2)``). Using numpy for uniform data would instead produce
    # ``(2, 2, 2)``, silently forking the result convention on whether the data
    # happened to be uniform. The general path produces the correct, consistent
    # convention for both uniform and ragged inputs.

    # expand_dims at axis_norm on each array, then concatenate
    expanded = [expand_dims(a, axis=axis_norm) for a in arrays]
    return concat(expanded, axis=axis_norm)
