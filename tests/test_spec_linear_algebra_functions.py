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
        len(transposed._impl),
    )
    assert ak.type(transposed._impl) == expected_type


def test_transpose_structure_and_values():
    arr = ragged.array([[[1, 2], [3]]])
    transposed = ragged.matrix_transpose(arr)
    expected_type = ak.types.ArrayType(
        ak.types.ListType(ak.types.ListType(ak.types.NumpyType("int64"))),
        len(transposed._impl),
    )
    assert ak.type(transposed._impl) == expected_type
    expected_array = [[[1, 3], [2]]]
    assert transposed.tolist() == expected_array


def test_can_transpose_unsorted():
    arr = ragged.array([[[1.1, 1.2, 1.3], [1.4, 1.5]], [[2.1, 2.2, 2.3]]])
    with pytest.raises(
        ValueError,
        match="Ragged dimension's lists must be sorted from longest to shortest, which is the only way that makes left-aligned ragged transposition possible.",
    ):
        ragged.matrix_transpose(arr)


def test_transpose_dtype_consistency():
    x = ragged.array([[[1.1, 3.3], [2.2]], [[4.4], [5.5]], [[6.6]], []])
    result = ragged.matrix_transpose(x)
    assert type(result._impl) is type(x._impl)
