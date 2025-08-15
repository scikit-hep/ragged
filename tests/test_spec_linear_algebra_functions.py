# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/linear_algebra_functions.html
"""

from __future__ import annotations

import awkward as ak
import pytest

import ragged


def test_existence():
    assert ragged.matmul is not None
    assert ragged.matrix_transpose is not None
    assert ragged.tensordot is not None
    assert ragged.vecdot is not None


def test_can_transpose_small():
    arr = ragged.array([[[1.1, 2.2, 3.3], [4.4]]])
    transpose = ragged.matrix_transpose(arr)
    expected_transpose = ragged.array([[[1.1, 4.4], [2.2], [3.3]]])
    assert ak.almost_equal(transpose._impl, expected_transpose._impl)


def test_can_transpose_with_empty():
    arr = ragged.array([[[1, 2], []]])
    expected_transpose = ragged.array([[[1], [2]]])
    assert ak.almost_equal(ragged.matrix_transpose(arr)._impl, expected_transpose._impl)


def test_can_transpose_allempty():
    arr = ragged.array([[[], []]])
    expected_transpose = ragged.array([[[], []]])
    assert ak.to_list(ragged.matrix_transpose(arr)._impl) == ak.to_list(
        expected_transpose._impl
    )


def test_can_transpose_shape_change():
    arr = ragged.array([[[1, 2], [3]]])
    transposed = ragged.matrix_transpose(arr)
    expected_type = ak.types.ArrayType(
        ak.types.ListType(ak.types.ListType(ak.types.NumpyType("int64"))),
        len(transposed),
    )
    assert ak.type(transposed._impl) == expected_type


def test_transpose_structure_and_values():
    arr = ragged.array([[[1, 2], [3]]])
    transposed = ragged.matrix_transpose(arr)
    expected_type = ak.types.ArrayType(
        ak.types.ListType(ak.types.ListType(ak.types.NumpyType("int64"))),
        len(transposed),
    )
    assert ak.type(transposed._impl) == expected_type
    expected_array = [[[1, 3], [2]]]
    assert transposed.tolist() == expected_array


def test_can_transpose_unsorted():
    arr = ragged.array([[[1.1, 1.2, 1.3], [1.4, 1.5]], [[2.1, 2.2, 2.3]]])
    message = "Ragged dimension's lists must be sorted from longest to shortest, which is the only way that makes left-aligned ragged transposition possible."
    with pytest.raises(ValueError, match=message):
        ragged.matrix_transpose(arr)


def test_transpose_dtype_consistency():
    x = ragged.array([[[1.1, 3.3], [2.2]], [[4.4], [5.5]], [[6.6]], []])
    result = ragged.matrix_transpose(x)
    assert result.dtype == x.dtype


def test_matmul_simple():
    x1 = ragged.array([[[2]], [[3]]])
    x2 = ragged.array([[[4]], [[5]]])
    output = ragged.array([[[8]], [[15]]])
    assert ak.to_list(ragged.matmul(x1, x2)._impl) == ak.to_list(output._impl)


def test_matmul_simple_more_dim():
    x1 = ragged.array([[[1, 0], [0, 1]], [[2, 3]]])
    x2 = ragged.array([[[1], [1]], [[4], [5]]])
    output = ragged.array([[[1], [1]], [[23]]])
    assert ak.to_list(ragged.matmul(x1, x2)._impl) == ak.to_list(output._impl)


def test_matmul_with_empty():
    x1 = ragged.array([[[], [1, 2]], [[]]])
    x2 = ragged.array([[[], [3]], [[]]])
    output = ragged.array([[[], [6]], [[]]])
    print("result type", type(ragged.matmul(x1, x2)))
    assert ak.to_list(ragged.matmul(x1, x2)._impl) == ak.to_list(output._impl)


def test_matmul_with_var_length():
    x1 = ragged.array([[[1.0, 2.0], [3.0]]])
    x2 = ragged.array([[[1.0], [2.0]]])
    output = ragged.array([[[5.0], [3.0]]])
    print("result type", type(ragged.matmul(x1, x2)))
    assert ak.to_list(ragged.matmul(x1, x2)._impl) == ak.to_list(output._impl)


# def test_matmul_1d_vectors():
#     x1 = ragged.array([1, 2, 3])
#     x2 = ragged.array([4, 5, 6])
#     output = ragged.array(1*4 + 2*5 + 3*6)  # scalar
#     assert ak.to_list(ragged.matmul(x1, x2)._impl) == ak.to_list(output._impl)


# def test_matmul_vector_matrix_promotion():
#     x1 = ragged.array([1, 2])
#     x2 = ragged.array([[3, 4], [5, 6]])
#     output = ragged.array([1*3+2*5, 1*4+2*6])
#     assert ak.to_list(ragged.matmul(x1, x2)._impl) == ak.to_list(output._impl)

#     x1 = ragged.array([[1, 2]])
#     x2 = ragged.array([3, 4])
#     output = ragged.array([1*3+2*4])
#     assert ak.to_list(ragged.matmul(x1, x2)._impl) == ak.to_list(output._impl)


# def test_matmul_batch_broadcasting():
#     x1 = ragged.array([[[1, 0], [0, 1]], [[2, 3], [4, 5]]])
#     x2 = ragged.array([[[1, 2], [3, 4]]])
#     output = ragged.array([[[1, 2], [3, 4]], [[11, 16], [19, 28]]])
#     assert ak.to_list(ragged.matmul(x1, x2)._impl) == ak.to_list(output._impl)


# def test_matmul_dimension_mismatch():
#     x1 = ragged.array([1, 2])
#     x2 = ragged.array([1, 2, 3])
#     with pytest.raises(ValueError, match="shape mismatch"):
#         ragged.matmul(x1, x2)

#     x1 = ragged.array([[1, 2]])
#     x2 = ragged.array([1, 2, 3])
#     with pytest.raises(ValueError, match="shape mismatch"):
#         ragged.matmul(x1, x2)


# def test_matmul_zero_dim_error():
#     x1 = ragged.array(5)
#     x2 = ragged.array([[1, 2]])
#     with pytest.raises(ValueError, match="zero-dimensional array"):
#         ragged.matmul(x1, x2)

#     x1 = ragged.array([[1, 2]])
#     x2 = ragged.array(5)
#     with pytest.raises(ValueError, match="zero-dimensional array"):
#         ragged.matmul(x1, x2)


def test_matmul_type_promotion():
    x1 = ragged.array([[1, 2]], dtype=int)
    x2 = ragged.array([[1.5, 2.5], [3.5, 4.5]], dtype=float)
    result = ragged.matmul(x1, x2)
    assert result.dtype.kind == "f"  # float output due to type promotion
