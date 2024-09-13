# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/set_functions.html
"""

from __future__ import annotations

import awkward as ak

import ragged


def test_existence():
    assert ragged.unique_all is not None
    assert ragged.unique_counts is not None
    assert ragged.unique_inverse is not None
    assert ragged.unique_values is not None


# unique_values tests
def test_can_take_list():
    arr = ragged.array([1, 2, 4, 3, 4, 5, 6, 20])
    expected_unique_values = ragged.array([1, 2, 3, 4, 5, 6, 20])
    unique_values = ragged.unique_values(arr)
    assert ak.to_list(expected_unique_values) == ak.to_list(unique_values)


def test_can_take_empty_arr():
    arr = ragged.array([])
    expected_unique_values = ragged.array([])
    unique_values = ragged.unique_values(arr)
    assert ak.to_list(expected_unique_values) == ak.to_list(unique_values)


def test_can_take_moredimensions():
    arr = ragged.array([[1, 2, 2, 3, 4], [5, 6]])
    expected_unique_values = ragged.array([1, 2, 3, 4, 5, 6])
    unique_values = ragged.unique_values(arr)
    assert ak.to_list(expected_unique_values) == ak.to_list(unique_values)


def test_can_take_1d_array():
    arr = ragged.array([5, 6, 7, 8, 8, 9, 1, 2, 3, 4, 10, 0, 15, 2])
    expected_unique_values = ragged.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15])
    assert ak.to_list(ragged.unique_values(arr)) == ak.to_list(expected_unique_values)


def test_can_take_scalar_int():
    arr = ragged.array(5)
    expected_unique_values = ragged.array(5)
    unique_values = ragged.unique_values(arr)
    assert unique_values == expected_unique_values


def test_can_take_scalar_float():
    arr = ragged.array(4.326)
    expected_unique_values = ragged.array(4.326)
    unique_values = ragged.unique_values(arr)
    assert unique_values == expected_unique_values


# unique_counts tests
def test_can_count_list():
    arr = ragged.array([1, 2, 4, 3, 4, 5, 6, 20])
    expected_unique_values = ragged.array([1, 2, 3, 4, 5, 6, 20])
    expected_unique_counts = ragged.array([1, 1, 1, 2, 1, 1, 1])
    unique_values, unique_counts = ragged.unique_counts(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_counts) == ak.to_list(expected_unique_counts)


def test_can_count_empty_arr():
    arr = ragged.array([])
    expected_unique_values = ragged.array([])
    expected_counts = ragged.array([])
    unique_values, unique_counts = ragged.unique_counts(arr)
    assert ak.to_list(expected_unique_values) == ak.to_list(unique_values)
    assert ak.to_list(expected_counts) == ak.to_list(unique_counts)


def test_can_count_simple_array():
    arr = ragged.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    expected_unique_values = ragged.array([1, 2, 3, 4])
    expected_counts = ragged.array([1, 2, 3, 4])
    unique_values, unique_counts = ragged.unique_counts(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_counts) == ak.to_list(expected_counts)


def test_can_count_normal_array():
    arr = ragged.array([[1, 2, 2], [3], [3, 3], [4, 4, 4], [4]])
    expected_unique_values = ragged.array([1, 2, 3, 4])
    expected_counts = ragged.array([1, 2, 3, 4])
    unique_values, unique_counts = ragged.unique_counts(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_counts) == ak.to_list(expected_counts)


def test_can_count_scalar_int():
    arr = ragged.array(5)
    expected_unique_values = ragged.array(5)
    expected_counts = ragged.array([1])
    unique_values, unique_counts = ragged.unique_counts(arr)
    assert unique_values == expected_unique_values
    assert unique_counts == expected_counts


def test_can_count_scalar_float():
    arr = ragged.array(4.326)
    expected_unique_values = ragged.array(4.326)
    expected_counts = ragged.array([1])
    unique_values, unique_counts = ragged.unique_counts(arr)
    assert unique_values == expected_unique_values
    assert unique_counts == expected_counts


# unique_inverse tests
def test_can_inverse_list():
    arr = ragged.array([1, 2, 4, 3, 4, 5, 6, 20])
    expected_unique_values = ragged.array([1, 2, 3, 4, 5, 6, 20])
    expected_inverse_indices = ragged.array([0, 1, 3, 2, 3, 4, 5, 6])
    unique_values, inverse_indices = ragged.unique_inverse(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(inverse_indices) == ak.to_list(expected_inverse_indices)


def test_can_inverse_empty_arr():
    arr = ragged.array([])
    expected_unique_values = ragged.array([])
    expected_inverse_indices = ragged.array([])
    unique_values, inverse_indices = ragged.unique_inverse(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(inverse_indices) == ak.to_list(expected_inverse_indices)


def test_can_inverse_simple_array():
    arr = ragged.array([[1, 2, 2], [3, 3, 3], [4, 4, 4, 4]])
    expected_unique_values = ragged.array([1, 2, 3, 4])
    expected_inverse_indices = ragged.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    unique_values, inverse_indices = ragged.unique_inverse(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(inverse_indices) == ak.to_list(expected_inverse_indices)


def test_can_inverse_normal_array():
    arr = ragged.array([[1, 2, 2], [3], [3, 3], [4, 4, 4], [4]])
    expected_unique_values = ragged.array([1, 2, 3, 4])
    expected_inverse_indices = ragged.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    unique_values, inverse_indices = ragged.unique_inverse(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(inverse_indices) == ak.to_list(expected_inverse_indices)


def test_can_inverse_scalar_int():
    arr = ragged.array(5)
    expected_unique_values = ragged.array(5)
    expected_inverse_indices = ragged.array([0])
    unique_values, inverse_indices = ragged.unique_inverse(arr)
    assert unique_values == expected_unique_values
    assert inverse_indices == expected_inverse_indices


def test_can_inverse_scalar_float():
    arr = ragged.array(4.326)
    expected_unique_values = ragged.array(4.326)
    expected_inverse_indices = ragged.array([0])
    unique_values, inverse_indices = ragged.unique_inverse(arr)
    assert unique_values == expected_unique_values
    assert inverse_indices == expected_inverse_indices


# unique_all tests
def test_can_all_list():
    arr = ragged.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    expected_unique_values = ragged.array([1, 2, 3, 4])
    expected_unique_indices = ragged.array([0, 1, 3, 6])
    expected_unique_inverse = ragged.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    expected_unique_counts = ragged.array([1, 2, 3, 4])
    unique_values, unique_indices, unique_inverse, unique_counts = ragged.unique_all(
        arr
    )
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_indices) == ak.to_list(expected_unique_indices)
    assert ak.to_list(unique_inverse) == ak.to_list(expected_unique_inverse)
    assert ak.to_list(unique_counts) == ak.to_list(expected_unique_counts)


def test_can_all_empty_arr():
    arr = ragged.array([])
    expected_unique_values = ragged.array([])
    expected_unique_indices = ragged.array([])
    expected_unique_inverse = ragged.array([])
    expected_unique_counts = ragged.array([])
    unique_values, unique_indices, unique_inverse, unique_counts = ragged.unique_all(
        arr
    )
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_indices) == ak.to_list(expected_unique_indices)
    assert ak.to_list(unique_inverse) == ak.to_list(expected_unique_inverse)
    assert ak.to_list(unique_counts) == ak.to_list(expected_unique_counts)


def test_can_all_normal_array():
    arr = ragged.array([[2, 2, 2], [3], [3, 5], [4, 4, 4], [4]])
    expected_unique_values = ragged.array([2, 3, 4, 5])
    expected_unique_indices = ragged.array([0, 3, 6, 5])
    expected_unique_inverse = ragged.array([0, 0, 0, 1, 1, 3, 2, 2, 2, 2])
    expected_unique_counts = ragged.array([3, 2, 4, 1])
    unique_values, unique_indices, unique_inverse, unique_counts = ragged.unique_all(
        arr
    )
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_indices) == ak.to_list(expected_unique_indices)
    assert ak.to_list(unique_inverse) == ak.to_list(expected_unique_inverse)
    assert ak.to_list(unique_counts) == ak.to_list(expected_unique_counts)


def test_can_all_scalar_int():
    arr = ragged.array(5)
    expected_unique_values = ragged.array(5)
    expected_unique_indices = ragged.array([0])
    expected_unique_inverse = ragged.array([0])
    expected_unique_counts = ragged.array([1])
    unique_values, unique_indices, unique_inverse, unique_counts = ragged.unique_all(
        arr
    )
    assert unique_values == expected_unique_values
    assert unique_indices == expected_unique_indices
    assert unique_inverse == expected_unique_inverse
    assert unique_counts == expected_unique_counts


def test_can_all_scalar_float():
    arr = ragged.array(4.326)
    expected_unique_values = ragged.array(4.326)
    expected_unique_indices = ragged.array([0])
    expected_unique_inverse = ragged.array([0])
    expected_unique_counts = ragged.array([1])
    unique_values, unique_indices, unique_inverse, unique_counts = ragged.unique_all(
        arr
    )
    assert unique_values == expected_unique_values
    assert unique_indices == expected_unique_indices
    assert unique_inverse == expected_unique_inverse
    assert unique_counts == expected_unique_counts
