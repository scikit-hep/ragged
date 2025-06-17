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
    expected_type = ak.types.ArrayType(1, ak.types.ListType(ak.types.ListType(ak.types.NumpyType("int64"))))
    assert ragged.matrix_transpose(arr)._impl.type == expected_type

def test_can_transpose_shape_change2():
    arr = ragged.array([[[1, 2], [3]]])
    transposed = ragged.matrix_transpose(arr)
    expected_type = ak.types.ArrayType(1,ak.types.ListType(ak.types.ListType(ak.types.NumpyType("int64"))))
    print(ak.type(transposed))
    print(repr(ak.type(transposed)))
    assert ak.types.equal(ak.type(transposed), expected_type)
    
def test_transpose_structure_and_values():
    arr = ragged.array([[[1, 2], [3]]]) 
    transposed = ragged.matrix_transpose(arr)

    expected_type = ak.types.ListType(ak.types.ListType(ak.types.NumpyType("int64")))
    assert transposed._impl.type == expected_type

    expected = [[[1, 3], [2]]]  # Transposing: [ [1, 2], [3] ] -> [ [1, 3], [2] ]
    assert transposed.to_list() == expected

def test_can_transpose_unsorted():
    arr = ragged.array([[1.1], [2, 3]]) 
    print(arr.ndim)
    print(arr[0].__len__())
    print(arr[1].__len__())
    with pytest.raises(ValueError, match="sorted descending"):
        ragged.matrix_transpose(arr)



