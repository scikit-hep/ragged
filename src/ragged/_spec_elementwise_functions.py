# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html
"""

from __future__ import annotations

import warnings

import numpy as np

from ._helper_functions import regularise_to_float
from ._spec_array_object import _box, _unbox, array


def abs(x: array, /) -> array:  # pylint: disable=W0622
    r"""
    Calculates the absolute value for each element `x_i` of the input array `x`.

    For real-valued input arrays, the element-wise result has the same
    magnitude as the respective element in `x` but has positive sign.

    For complex floating-point operands, the complex absolute value is known as
    the norm, modulus, or magnitude and, for a complex number `z = a + bj` is
    computed as

    $$\operatorname{abs}(z) = \sqrt{a^2 + b^2}$$

    Args:
        x: Input array.

    Returns:
        An array containing the absolute value of each element in `x`. If `x`
        has a real-valued data type, the returned array has the same data type
        as `x`. If `x` has a complex floating-point data type, the returned
        array has a real-valued floating-point data type whose precision
        matches the precision of `x` (e.g., if `x` is `complex128`, then the
        returned array has a `float64` data type).

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.abs.html
    """

    return _box(type(x), np.absolute(*_unbox(x)))


def acos(x: array, /) -> array:
    r"""
    Calculates an approximation of the principal value of the inverse cosine
    for each element `x_i` of the input array `x`.

    Each element-wise result is expressed in radians.

    The principal value of the arc cosine of a complex number `z` is

    $$\operatorname{acos}(z) = \frac{1}{2}\pi + j\ \ln(zj + \sqrt{1-z^2})$$

    For any `z`,

    $$\operatorname{acos}(z) = \pi - \operatorname{acos}(-z)$$

    Args:
        x: Input array.

    Returns:
        An array containing the inverse cosine of each element in `x`. The
        returned array has a floating-point data type determined by type
        promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.acos.html
    """

    return _box(type(x), np.arccos(*_unbox(x)))


def acosh(x: array, /) -> array:
    r"""
    Calculates an approximation to the inverse hyperbolic cosine for each
    element `x_i` of the input array `x`.

    The principal value of the inverse hyperbolic cosine of a complex number
    `z` is

    $$\operatorname{acosh}(z) = \ln(z + \sqrt{z+1}\sqrt{z-1})$$

    For any `z`,

    $$\operatorname{acosh}(z) = \frac{\sqrt{z-1}}{\sqrt{1-z}}\operatorname{acos}(z)$$

    or simply

    $$\operatorname{acosh}(z) = j\ \operatorname{acos}(z)$$

    in the upper half of the complex plane.

    Args:
        x: Input array whose elements each represent the area of a hyperbolic
            sector.

    Returns:
        An array containing the inverse hyperbolic cosine of each element in
        `x`. The returned array has a floating-point data type determined by
        type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.acosh.html
    """

    return _box(type(x), np.arccosh(*_unbox(x)))


def add(x1: array, x2: array, /) -> array:
    """
    Calculates the sum for each element `x1_i` of the input array `x1` with the
    respective element `x2_i` of the input array `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise sums. The returned array has a
        data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.add.html
    """

    return _box(type(x1), np.add(*_unbox(x1, x2)))


def asin(x: array, /) -> array:
    r"""
    Calculates an approximation of the principal value of the inverse sine for
    each element `x_i` of the input array `x`.

    Each element-wise result is expressed in radians.

    The principal value of the arc sine of a complex number `z` is

    $$\operatorname{asin}(z) = -j\ \ln(zj + \sqrt{1-z^2})$$

    For any `z`,

    $$\operatorname{asin}(z) = \operatorname{acos}(-z) - \frac{\pi}{2}$$

    Args:
        x: Input array.

    Returns:
        An array containing the inverse sine of each element in `x`. The
        returned array has a floating-point data type determined by type
        promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.asin.html
    """

    return _box(type(x), np.arcsin(*_unbox(x)))


def asinh(x: array, /) -> array:
    r"""
    Calculates an approximation to the inverse hyperbolic sine for each element
    `x_i` in the input array `x`.

    The principal value of the inverse hyperbolic sine of a complex number `z`
    is

    $$\operatorname{asinh}(z) = \ln(z + \sqrt{1+z^2})$$

    For any `z`,

    $$\operatorname{asinh}(z) = \frac{\operatorname{asin}(zj)}{j}$$

    Args:
        x: Input array whose elements each represent the area of a hyperbolic
        sector.

    Returns:
        An array containing the inverse hyperbolic sine of each element in `x`.
        The returned array has a floating-point data type determined by type
        promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.asinh.html
    """

    return _box(type(x), np.arcsinh(*_unbox(x)))


def atan(x: array, /) -> array:
    r"""
    Calculates an approximation of the principal value of the inverse tangent
    for each element `x_i` of the input array `x`.

    Each element-wise result is expressed in radians.

    The principal value of the inverse tangent of a complex number `z` is

    $$\operatorname{atan}(z) = -\frac{\ln(1 - zj) - \ln(1 + zj)}{2}j$$

    Args:
        x: Input array. Should have a floating-point data type.

    Returns:
        An array containing the inverse tangent of each element in `x`. The
        returned array has a floating-point data type determined by type
        promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.atan.html
    """

    return _box(type(x), np.arctan(*_unbox(x)))


def atan2(x1: array, x2: array, /) -> array:
    """
    Calculates an approximation of the inverse tangent of the quotient `x1/x2`,
    having domain `[-infinity, +infinity] \u00d7 [-infinity, +infinity]` (where
    the `\u00d7` notation denotes the set of ordered pairs of elements
    `(x1_i, x2_i)`) and codomain `[-π, +π]`, for each pair of elements
    `(x1_i, x2_i)` of the input arrays `x1` and `x2`, respectively. Each
    element-wise result is expressed in radians.

    The mathematical signs of `x1_i` and `x2_i` determine the quadrant of each
    element-wise result. The quadrant (i.e., branch) is chosen such that each
    element-wise result is the signed angle in radians between the ray ending
    at the origin and passing through the point `(1, 0)` and the ray ending at
    the origin and passing through the point `(x2_i, x1_i)`.

    Note the role reversal: the "y-coordinate" is the first function parameter;
    the "x-coordinate" is the second function parameter. The parameter order is
    intentional and traditional for the two-argument inverse tangent function
    where the y-coordinate argument is first and the x-coordinate argument is
    second.

    By IEEE 754 convention, the inverse tangent of the quotient `x1/x2` is
    defined for `x2_i` equal to positive or negative zero and for either or
    both of `x1_i` and `x2_i` equal to positive or negative `infinity`.

    Args:
        x1: Input array corresponding to the y-coordinates.
        x2: Input array corresponding to the x-coordinates. Must be
        broadcastable with `x1`.

    Returns:
        An array containing the inverse tangent of the quotient `x1/x2`. The
        returned array has a real-valued floating-point data type determined by
        type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.atan2.html
    """

    return _box(type(x1), np.arctan2(*_unbox(x1, x2)))


def atanh(x: array, /) -> array:
    r"""
    Calculates an approximation to the inverse hyperbolic tangent for each
    element `x_i` of the input array `x`.

    The principal value of the inverse hyperbolic tangent of a complex number
    `z` is

    $$\operatorname{atanh}(z) = \frac{\ln(1+z)-\ln(z-1)}{2}$$

    For any `z`,

    $$\operatorname{atanh}(z) = \frac{\operatorname{atan}(zj)}{j}$$

    Args:
        x: Input array whose elements each represent the area of a hyperbolic
        sector.

    Returns:
        An array containing the inverse hyperbolic tangent of each element in
        `x`. The returned array has a floating-point data type determined by
        type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.atanh.html
    """

    return _box(type(x), np.arctanh(*_unbox(x)))


def bitwise_and(x1: array, x2: array, /) -> array:
    """
    Computes the bitwise AND of the underlying binary representation of each
    element `x1_i` of the input array `x1` with the respective element `x2_i`
    of the input array `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.bitwise_and.html
    """

    return _box(type(x1), np.bitwise_and(*_unbox(x1, x2)))


def bitwise_invert(x: array, /) -> array:
    """
    Inverts (flips) each bit for each element `x_i` of the input array `x`.

    Args:
        x: Input array.

    Returns:
        An array containing the element-wise results. The returned array has
        the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.bitwise_invert.html
    """

    return _box(type(x), np.invert(*_unbox(x)))


def bitwise_left_shift(x1: array, x2: array, /) -> array:
    """
    Shifts the bits of each element `x1_i` of the input array `x1` to the left
    by appending `x2_i` (i.e., the respective element in the input array `x2`)
    zeros to the right of `x1_i`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`. Each element
            must be greater than or equal to 0.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.bitwise_left_shift.html
    """

    return _box(type(x1), np.left_shift(*_unbox(x1, x2)))


def bitwise_or(x1: array, x2: array, /) -> array:
    """
    Computes the bitwise OR of the underlying binary representation of each
    element `x1_i` of the input array `x1` with the respective element `x2_i`
    of the input array `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.bitwise_or.html
    """

    return _box(type(x1), np.bitwise_or(*_unbox(x1, x2)))


def bitwise_right_shift(x1: array, x2: array, /) -> array:
    """
    Shifts the bits of each element `x1_i` of the input array `x1` to the right
    according to the respective element `x2_i` of the input array `x2`.

    This operation is equivalent to floor division by a power of two.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`. Each element
        must be greater than or equal to 0.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.bitwise_right_shift.html
    """

    return _box(type(x1), np.right_shift(*_unbox(x1, x2)))


def bitwise_xor(x1: array, x2: array, /) -> array:
    """
    Computes the bitwise XOR of the underlying binary representation of each
    element `x1_i` of the input array `x1` with the respective element `x2_i`
    of the input array `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.bitwise_xor.html
    """

    return _box(type(x1), np.bitwise_xor(*_unbox(x1, x2)))


def ceil(x: array, /) -> array:
    """
    Rounds each element `x_i` of the input array `x` to the smallest (i.e.,
    closest to `-infinity`) integer-valued number that is not less than `x_i`.

    Args:
        x: Input array.

    Returns:
        An array containing the rounded result for each element in `x`. The
        returned array has the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.ceil.html
    """

    return _box(type(x), np.ceil(*_unbox(x)), dtype=regularise_to_float(x.dtype))


def conj(x: array, /) -> array:
    """
    Returns the complex conjugate for each element `x_i` of the input array
    `x`.

    For complex numbers of the form

    $$a + bj$$

    the complex conjugate is defined as

    $$a - bj$$

    Hence, the returned complex conjugates is computed by negating the
    imaginary component of each element `x_i`.

    Args:
        x: Input array.

    Returns:
        An array containing the element-wise results. The returned array has
        the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.conj.html
    """

    return _box(type(x), np.conjugate(*_unbox(x)))


def cos(x: array, /) -> array:
    r"""
    Calculates an approximation to the cosine for each element `x_i` of the
    input array `x`.

    Each element `x_i` is assumed to be expressed in radians.

    Args:
        x: Input array whose elements are each expressed in radians.

    Returns:
        An array containing the cosine of each element in `x`. The returned
        array has a floating-point data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.cos.html
    """

    return _box(type(x), np.cos(*_unbox(x)))


def cosh(x: array, /) -> array:
    r"""
    Calculates an approximation to the hyperbolic cosine for each element `x_i`
    in the input array `x`.

    The mathematical definition of the hyperbolic cosine is

    $$\operatorname{cosh}(x) = \frac{e^x + e^{-x}}{2}$$

    Args:
        x: Input array whose elements each represent a hyperbolic angle.

    Returns:
        An array containing the hyperbolic cosine of each element in `x`. The
        returned array has a floating-point data type determined by type
        promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.cosh.html
    """

    return _box(type(x), np.cosh(*_unbox(x)))


def divide(x1: array, x2: array, /) -> array:
    r"""
    Calculates the division of each element `x1_i` of the input array `x1` with
    the respective element `x2_i` of the input array `x2`.

    Args:
        x1: Dividend input array.
        x2: Divisor input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        floating-point data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.divide.html
    """

    return _box(type(x1), np.divide(*_unbox(x1, x2)))


def equal(x1: array, x2: array, /) -> array:
    r"""
    Computes the truth value of `x1_i == x2_i` for each element `x1_i` of the
    input array `x1` with the respective element `x2_i` of the input array
    `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type of `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.equal.html
    """

    return _box(type(x1), np.equal(*_unbox(x1, x2)))


def exp(x: array, /) -> array:
    """
    Calculates an approximation to the exponential function for each element
    `x_i` of the input array `x` (`e` raised to the power of `x_i`, where `e`
    is the base of the natural logarithm).

    Args:
        x: Input array.

    Returns:
        An array containing the evaluated exponential function result for each
        element in `x`. The returned array has a floating-point data type
        determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.exp.html
    """

    return _box(type(x), np.exp(*_unbox(x)))


def expm1(x: array, /) -> array:
    """
    Calculates an approximation to `exp(x)-1` for each element `x_i` of the
    input array `x`.

    The purpose of this function is to calculate `exp(x)-1.0` more accurately
    when `x` is close to zero.

    Args:
        x: Input array.

    Returns:
        An array containing the evaluated result for each element in `x`. The
        returned array has a floating-point data type determined by type
        promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.expm1.html
    """

    return _box(type(x), np.expm1(*_unbox(x)))


def floor(x: array, /) -> array:
    """
    Rounds each element `x_i` of the input array `x` to the greatest (i.e.,
    closest to `+infinity`) integer-valued number that is not greater than
    `x_i`.

    Args:
        x: Input array.

    Returns:
        An array containing the rounded result for each element in `x`. The
        returned array must have the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.floor.html
    """

    return _box(type(x), np.floor(*_unbox(x)), dtype=regularise_to_float(x.dtype))


def floor_divide(x1: array, x2: array, /) -> array:
    r"""
    Rounds the result of dividing each element `x1_i` of the input array `x1`
    by the respective element `x2_i` of the input array `x2` to the greatest
    (i.e., closest to `+infinity`) integer-value number that is not greater
    than the division result.

    Args:
        x1: Dividend input array.
        x2: Divisor input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.floor_divide.html
    """

    return _box(type(x1), np.floor_divide(*_unbox(x1, x2)))


def greater(x1: array, x2: array, /) -> array:
    """
    Computes the truth value of `x1_i > x2_i` for each element `x1_i` of the
    input array `x1` with the respective element `x2_i` of the input array
    `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type of `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.greater.html
    """

    return _box(type(x1), np.greater(*_unbox(x1, x2)))


def greater_equal(x1: array, x2: array, /) -> array:
    """
    Computes the truth value of `x1_i >= x2_i` for each element `x1_i` of the
    input array `x1` with the respective element `x2_i` of the input array
    `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type of `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.greater_equal.html
    """

    return _box(type(x1), np.greater_equal(*_unbox(x1, x2)))


def imag(x: array, /) -> array:
    """
    Returns the imaginary component of a complex number for each element `x_i`
    of the input array `x`.

    Args:
        x: Input array.

    Returns:
        An array containing the element-wise results. The returned array has a
        floating-point data type with the same floating-point precision as `x`
        (e.g., if `x` is `complex64`, the returned array has the floating-point
        data type `float32`).

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.imag.html
    """

    (a,) = _unbox(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _box(
            type(x),
            (a - np.conjugate(a)) / 2j,
            dtype=np.dtype(f"f{x.dtype.itemsize // 2}"),
        )


def isfinite(x: array, /) -> array:
    """
    Tests each element `x_i` of the input array `x` to determine if finite.

    Args:
        x: Input array.

    Returns:
        An array containing test results. The returned array has a data type of
        `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isfinite.html
    """

    return _box(type(x), np.isfinite(*_unbox(x)))


def isinf(x: array, /) -> array:
    """
    Tests each element `x_i` of the input array `x` to determine if equal to
    positive or negative infinity.

    Args:
        x: Input array.

    Returns:
        An array containing test results. The returned array has a data type of
        `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isinf.html
    """

    return _box(type(x), np.isinf(*_unbox(x)))


def isnan(x: array, /) -> array:
    """
    Tests each element `x_i` of the input array `x` to determine whether the
    element is `NaN`.

    Args:
        x: Input array.

    Returns:
        An array containing test results. The returned array has a data type of
        `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isnan.html
    """

    return _box(type(x), np.isnan(*_unbox(x)))


def less(x1: array, x2: array, /) -> array:
    """
    Computes the truth value of `x1_i < x2_i` for each element `x1_i` of the
    input array `x1` with the respective element `x2_i` of the input array
    `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type of `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.less.html
    """

    return _box(type(x1), np.less(*_unbox(x1, x2)))


def less_equal(x1: array, x2: array, /) -> array:
    """
    Computes the truth value of `x1_i <= x2_i` for each element `x1_i` of the
    input array `x1` with the respective element `x2_i` of the input array
    `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type of `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.less_equal.html
    """

    return _box(type(x1), np.less_equal(*_unbox(x1, x2)))


def log(x: array, /) -> array:
    r"""
    Calculates an approximation to the natural (base `e`) logarithm for each
    element `x_i` of the input array `x`.

    Args:
        x: Input array.

    Returns:
        An array containing the evaluated natural logarithm for each element in
        `x`. The returned array has a floating-point data type determined by
        type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.log.html
    """

    return _box(type(x), np.log(*_unbox(x)))


def log1p(x: array, /) -> array:
    r"""
    Calculates an approximation to `log(1+x)`, where `log` refers to the
    natural (base `e`) logarithm, for each element `x_i` of the input array
    `x`.

    The purpose of this function is to calculate `log(1+x)` more accurately
    when `x` is close to zero.

    Args:
        x: Input array.

    Returns:
        An array containing the evaluated result for each element in `x`. The
        returned array has a floating-point data type determined by type
        promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.log1p.html
    """

    return _box(type(x), np.log1p(*_unbox(x)))


def log2(x: array, /) -> array:
    r"""
    Calculates an approximation to the base 2 logarithm for each element `x_i`
    of the input array `x`.

    Args:
        x: Input array.

    Returns:
        An array containing the evaluated base 2 logarithm for each element in
        `x`. The returned array has a floating-point data type determined by
        type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.log2.html
    """

    return _box(type(x), np.log2(*_unbox(x)))


def log10(x: array, /) -> array:
    r"""
    Calculates an approximation to the base 10 logarithm for each element `x_i`
    of the input array `x`.

    Args:
        x: Input array.

    Returns:
        An array containing the evaluated base 10 logarithm for each element in
        `x`. The returned array has a floating-point data type determined by
        type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.log10.html
    """

    return _box(type(x), np.log10(*_unbox(x)))


def logaddexp(x1: array, x2: array, /) -> array:
    """
    Calculates the logarithm of the sum of exponentiations
    `log(exp(x1) + exp(x2))` for each element `x1_i` of the input array `x1`
    with the respective element `x2_i` of the input array `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        real-valued floating-point data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.logaddexp.html
    """

    return _box(type(x1), np.logaddexp(*_unbox(x1, x2)))


def logical_and(x1: array, x2: array, /) -> array:
    """
    Computes the logical AND for each element `x1_i` of the input array `x1`
    with the respective element `x2_i` of the input array `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type of `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.logical_and.html
    """

    return _box(type(x1), np.logical_and(*_unbox(x1, x2)))


def logical_not(x: array, /) -> array:
    """
    Computes the logical NOT for each element `x_i` of the input array `x`.

    Args:
        x: Input array.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type of `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.logical_not.html
    """

    return _box(type(x), np.logical_not(*_unbox(x)))


def logical_or(x1: array, x2: array, /) -> array:
    """
    Computes the logical OR for each element `x1_i` of the input array `x1`
    with the respective element `x2_i` of the input array `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type of `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.logical_or.html
    """

    return _box(type(x1), np.logical_or(*_unbox(x1, x2)))


def logical_xor(x1: array, x2: array, /) -> array:
    """
    Computes the logical XOR for each element `x1_i` of the input array `x1`
    with the respective element `x2_i` of the input array `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type of `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.logical_xor.html
    """

    return _box(type(x1), np.logical_xor(*_unbox(x1, x2)))


def multiply(x1: array, x2: array, /) -> array:
    r"""
    Calculates the product for each element `x1_i` of the input array `x1` with
    the respective element `x2_i` of the input array `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise products. The returned array has a
        data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.multiply.html
    """

    return _box(type(x1), np.multiply(*_unbox(x1, x2)))


def negative(x: array, /) -> array:
    """
    Computes the numerical negative of each element `x_i` (i.e., `y_i = -x_i`)
    of the input array `x`.

    Args:
        x: Input array.

    Returns:
        An array containing the evaluated result for each element in `x`. The
        returned array has a data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.negative.html
    """

    return _box(type(x), np.negative(*_unbox(x)))


def not_equal(x1: array, x2: array, /) -> array:
    """
    Computes the truth value of `x1_i != x2_i` for each element `x1_i` of the
    input array `x1` with the respective element `x2_i` of the input array
    `x2`.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type of `bool`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.not_equal.html
    """

    return _box(type(x1), np.not_equal(*_unbox(x1, x2)))


def positive(x: array, /) -> array:
    """
    Computes the numerical positive of each element `x_i` (i.e., `y_i = +x_i`)
    of the input array `x`.

    Args:
        x: Input array.

    Returns:
        An array containing the evaluated result for each element in `x`. The
        returned array has the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.positive.html
    """

    return _box(type(x), np.positive(*_unbox(x)))


def pow(x1: array, x2: array, /) -> array:  # pylint: disable=W0622
    r"""
    Calculates an approximation of exponentiation by raising each element
    `x1_i` (the base) of the input array `x1` to the power of `x2_i` (the
    exponent), where `x2_i` is the corresponding element of the input array
    `x2`.

    Args:
        x1: First input array whose elements correspond to the exponentiation
        base.
        x2: Second input array whose elements correspond to the exponentiation
        exponent. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. The returned array has a
        data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.pow.html
    """

    return _box(type(x1), np.power(*_unbox(x1, x2)))


def real(x: array, /) -> array:
    """
    Returns the real component of a complex number for each element `x_i` of
    the input array `x`.

    Args:
        x: Input array.

    Returns:
        An array containing the element-wise results. The returned array has a
        floating-point data type with the same floating-point precision as `x`
        (e.g., if `x` is `complex64`, the returned array has the floating-point
        data type `float32`).

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.real.html
    """

    (a,) = _unbox(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _box(
            type(x),
            (a + np.conjugate(a)) / 2,
            dtype=np.dtype(f"f{x.dtype.itemsize // 2}"),
        )


def remainder(x1: array, x2: array, /) -> array:
    """
    Returns the remainder of division for each element `x1_i` of the input
    array `x1` and the respective element `x2_i` of the input array `x2`.

    This function is equivalent to the Python modulus operator `x1_i % x2_i`.

    Args:
        x1: Dividend input array.
        x2: Divisor input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise results. Each element-wise result
        has the same sign as the respective element `x2_i`. The returned array
        has a data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.remainder.html
    """

    return _box(type(x1), np.remainder(*_unbox(x1, x2)))


def round(x: array, /) -> array:  # pylint: disable=W0622
    """
    Rounds each element `x_i` of the input array `x` to the nearest
    integer-valued number.

    For complex floating-point operands, real and imaginary components are
    independently rounded to the nearest integer-valued number.

    Rounded real and imaginary components are equal to their equivalent rounded
    real-valued floating-point counterparts (i.e., for complex-valued `x`,
    `real(round(x))` must equal `round(real(x)))` and `imag(round(x))` equals
    `round(imag(x))`).

    Args:
        x: Input array.

    Returns:
        An array containing the rounded result for each element in `x`. The
        returned array has the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.round.html
    """

    (a,) = _unbox(x)
    if x.dtype in (np.complex64, np.complex128):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a_conj = np.conjugate(a)
            dt = np.dtype(f"f{x.dtype.itemsize // 2}")
            re = _box(type(x), (a + a_conj) / 2, dtype=dt)
            im = _box(type(x), (a - a_conj) / 2j, dtype=dt)
            return add(round(re), multiply(round(im), array(1j, device=x.device)))

    else:
        frac, whole = np.modf(a)
        abs_frac = np.absolute(frac)
        return _box(
            type(x),
            whole
            + ((abs_frac == 0.5) * (whole % 2 != 0) + (abs_frac > 0.5)) * np.sign(frac),
        )


def sign(x: array, /) -> array:
    r"""
    Returns an indication of the sign of a number for each element `x_i` of the
    input array `x`.

    The sign function (also known as the **signum function**) of a number $x_i$
    is defined as

    $$\operatorname{sign}(x_i) = \begin{cases} 0 & \textrm{if } x_i = 0 \\ \frac{x}{|x|} & \textrm{otherwise} \end{cases}$$

    where $|x_i|$ is the absolute value of $x_i$.

    Args:
        x: Input array.

    Returns:
        An array containing the evaluated result for each element in `x`. The
        returned array has the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.sign.html
    """

    return _box(type(x), np.sign(*_unbox(x)))


def sin(x: array, /) -> array:
    r"""
    Calculates an approximation to the sine for each element `x_i` of the input
    array `x`.

    Each element `x_i` is assumed to be expressed in radians.

    Args:
        x: Input array whose elements are each expressed in radians.

    Returns:
        An array containing the sine of each element in `x`. The returned array
        has a floating-point data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.sin.html
    """

    return _box(type(x), np.sin(*_unbox(x)))


def sinh(x: array, /) -> array:
    r"""
    Calculates an approximation to the hyperbolic sine for each element `x_i`
    of the input array `x`.

    The mathematical definition of the hyperbolic sine is

    $$\operatorname{sinh}(x) = \frac{e^x - e^{-x}}{2}$$

    Args:
        x: Input array whose elements each represent a hyperbolic angle.

    Returns:
        An array containing the hyperbolic sine of each element in `x`. The
        returned array has a floating-point data type determined by type
        promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.sinh.html
    """

    return _box(type(x), np.sinh(*_unbox(x)))


def square(x: array, /) -> array:
    r"""
    Squares each element `x_i` of the input array `x`.

    The square of a number `x_i` is defined as

    $$x_i^2 = x_i \cdot x_i$$

    Args:
        x: Input array.

    Returns:
        An array containing the evaluated result for each element in `x`. The
        returned array has a data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.square.html
    """

    return _box(type(x), np.square(*_unbox(x)))


def sqrt(x: array, /) -> array:
    r"""
    Calculates the principal square root for each element `x_i` of the input
    array `x`.

    Args:
        x: Input array.

    Returns:
        An array containing the square root of each element in `x`. The
        returned array has a floating-point data type determined by type
        promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.sqrt.html
    """

    return _box(type(x), np.sqrt(*_unbox(x)))


def subtract(x1: array, x2: array, /) -> array:
    """
    Calculates the difference for each element `x1_i` of the input array `x1`
    with the respective element `x2_i` of the input array `x2`.

    The result of `x1_i - x2_i` is the same as `x1_i + (-x2_i)` and is governed
    by the same floating-point rules as addition.

    Args:
        x1: First input array.
        x2: Second input array. Must be broadcastable with `x1`.

    Returns:
        An array containing the element-wise differences. The returned array
        has a data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.subtract.html
    """

    return _box(type(x1), np.subtract(*_unbox(x1, x2)))


def tan(x: array, /) -> array:
    r"""
    Calculates an approximation to the tangent for each element `x_i` of the
    input array `x`.

    Each element `x_i` is assumed to be expressed in radians.

    Args:
        x: Input array whose elements are expressed in radians.

    Returns:
        An array containing the tangent of each element in `x`. The returned
        array has a floating-point data type determined by type promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.tan.html
    """

    return _box(type(x), np.tan(*_unbox(x)))


def tanh(x: array, /) -> array:
    r"""
    Calculates an approximation to the hyperbolic tangent for each element
    `x_i` of the input array `x`.

    The mathematical definition of the hyperbolic tangent is

    $$\begin{align} \operatorname{tanh}(x) &= \frac{\operatorname{sinh}(x)}{\operatorname{cosh}(x)} \\ &= \frac{e^x - e^{-x}}{e^x + e^{-x}} \end{align}$$

    where $\operatorname{sinh}(x)$ is the hyperbolic sine and
    $\operatorname{cosh}(x)$ is the hyperbolic cosine.

    Args:
        x: Input array whose elements each represent a hyperbolic angle.

    Returns:
        An array containing the hyperbolic tangent of each element in `x`. The
        returned array has a floating-point data type determined by type
        promotion.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.tanh.html
    """

    return _box(type(x), np.tanh(*_unbox(x)))


def trunc(x: array, /) -> array:
    """
    Rounds each element `x_i` of the input array `x` to the nearest
    integer-valued number that is closer to zero than `x_i`.

    Args:
        x: Input array.

    Returns:
        An array containing the rounded result for each element in `x`. The
        returned array has the same data type as `x`.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.trunc.html
    """

    return _box(type(x), np.trunc(*_unbox(x)))
