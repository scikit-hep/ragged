# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html
"""

from __future__ import annotations

import numbers

import awkward as ak
import numpy as np

from ._spec_array_object import _box, _unbox, array


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


def broadcast_to(x: array, /, shape: tuple[int, ...]) -> array:
    """
    Broadcasts an array to a specified shape.

    Args:
        x: Array to broadcast.
        shape: Array shape. Must be compatible with `x`. If the array is
        incompatible with the specified shape, the function raises an
        exception.

    Returns:
        An array having a specified shape. Must have the same data type as x.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.broadcast_to.html
    """

    x  # noqa: B018, pylint: disable=W0104
    shape  # noqa: B018, pylint: disable=W0104
    raise NotImplementedError("TODO 115")  # noqa: EM101


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

    x  # noqa: B018, pylint: disable=W0104
    axis  # noqa: B018, pylint: disable=W0104
    raise NotImplementedError("TODO 118")  # noqa: EM101


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

    x  # noqa: B018, pylint: disable=W0104
    axes  # noqa: B018, pylint: disable=W0104
    raise NotImplementedError("TODO 119")  # noqa: EM101


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

    x  # noqa: B018, pylint: disable=W0104
    shape  # noqa: B018, pylint: disable=W0104
    copy  # noqa: B018, pylint: disable=W0104
    raise NotImplementedError("TODO 120")  # noqa: EM101


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

    x  # noqa: B018, pylint: disable=W0104
    shift  # noqa: B018, pylint: disable=W0104
    axis  # noqa: B018, pylint: disable=W0104
    raise NotImplementedError("TODO 121")  # noqa: EM101


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

    arrays  # noqa: B018, pylint: disable=W0104
    axis  # noqa: B018, pylint: disable=W0104
    raise NotImplementedError("TODO 123")  # noqa: EM101
