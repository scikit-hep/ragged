# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/linear_algebra_functions.html
"""

from __future__ import annotations

import pytest

import awkward as ak
import ragged


def test_existence():
    assert ragged.matmul is not None
    assert ragged.matrix_transpose is not None
    assert ragged.tensordot is not None
    assert ragged.vecdot is not None

def test_can_transpose_small():
    arr = ragged.array([[[1.1, 2.2, 3.3], [4.4]]])
    transpose = ragged.matrix_transpose(arr)
    expected_transpose= ragged.array([[[1.1, 4.4], [2.2], [3.3]]])
    assert ak.almost_equal(transpose._impl, expected_transpose._impl)

def test_can_transpose_with_empty():
    arr = ragged.array([[[1, 2], []]])
    expected_transpose = ragged.array([[[1], [2]]])
    assert ak.almost_equal(ragged.matrix_transpose(arr)._impl, expected_transpose._impl)

def test_can_transpose_allempty():
    arr = ragged.array([[[], []]])
    expected_transpose = ragged.array([[]])
    assert ak.almost_equal(ragged.matrix_transpose(arr)._impl, expected_transpose._impl)

def test_can_transpose_stack():
    arr = ragged.array([[[1.1, 2.2], [3.3]],
    [[4.4, 5.5, 6.6]],[]])
    expected_transpose = ragged.array([[[1.1, 3.3], [2.2]],
    [[4.4], [5.5], [6.6]],[]])
    assert ak.almost_equal(ragged.matrix_transpose(arr)._impl, expected_transpose._impl)

def test_can_transpose_shape_change():
    arr = ragged.array([[[1, 2], [3]]])
    assert ragged.matrix_transpose(arr)._impl.type == ak.type.ArrayType(1, ak.types.ListType(ak.types.ListType(ak.types.PrimitiveType("int64"))))

def test_can_transpose_unsorted():
    arr = ragged.array([[[1], [2, 3]]]) 
    with pytest.raises(ValueError, match="sorted descending"):
        ragged.matrix_transpose(arr)



