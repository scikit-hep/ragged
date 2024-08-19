# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/set_functions.html
"""

from __future__ import annotations

import re

import awkward as ak
import pytest

import ragged


def test_existence():
    assert ragged.unique_all is not None
    assert ragged.unique_counts is not None
    assert ragged.unique_inverse is not None
    assert ragged.unique_values is not None


# unique_values tests
def test_can_take_none():
    with pytest.raises(TypeError):
        assert ragged.unique_values(ragged.array(None)) is None


def test_can_take_list():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "the truth value of an array whose length is not 1 is ambiguous;"
        ),
    ):
        assert ragged.unique_values(
            ragged.array([1, 2, 4, 3, 4, 5, 6, 20])
        ) == ragged.array([1, 2, 3, 4, 5, 6, 20])


def test_can_take_empty_arr():
    with pytest.raises(TypeError):
        assert ragged.unique_values(ragged.array([])) == ragged.array([])


def test_can_take_moredimensions():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "the truth value of an array whose length is not 1 is ambiguous;"
        ),
    ):
        assert ragged.unique_values(ragged.array([[1, 2, 3, 4], [5, 6]]))


def test_can_take_1d_array():
    arr = ragged.array([5, 6, 7, 8, 8, 9, 1, 2, 3, 4, 10, 0, 15, 2])
    expected_unique_values = ragged.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15])
    assert ak.to_list(ragged.unique_values(arr)) == ak.to_list(expected_unique_values)


# unique_counts tests
def test_can_count_none():
    with pytest.raises(TypeError):
        assert ragged.unique_counts(ragged.array(None)) is None


def test_can_count_list():
    with pytest.raises(TypeError):
        assert ragged.unique_counts(
            ragged.array([1, 2, 4, 3, 4, 5, 6, 20])
        ) == ragged.array([1, 2, 3, 4, 5, 6, 20]), ragged.array([1, 1, 2, 1, 1, 1, 1])


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


def test_can_count_scalar():
    arr = ragged.array([5])
    expected_unique_values = ragged.array([5])
    expected_counts = ragged.array([1])
    unique_values, unique_counts = ragged.unique_counts(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_counts) == ak.to_list(expected_counts)


# unique_inverse tests
def test_can_inverse_none():
    with pytest.raises(TypeError):
        assert ragged.unique_inverse(ragged.array(None)) is None


def test_can_inverse_list():
    with pytest.raises(TypeError):
        assert ragged.unique_inverse(
            ragged.array([1, 2, 4, 3, 4, 5, 6, 20])
        ) == ragged.array([1, 2, 3, 4, 5, 6, 20]), ragged.array(
            [0, 1, 3, 2, 3, 4, 5, 6]
        )


def test_can_take_simple_array():
    arr = ragged.array([[1, 2, 2], [3, 3, 3], [4, 4, 4, 4]])
    expected_unique_values = ragged.array([1, 2, 3, 4])
    expected_inverse_indices = ragged.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    unique_values, inverse_indices = ragged.unique_inverse(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(inverse_indices) == ak.to_list(expected_inverse_indices)


def test_can_take_normal_array():
    arr = ragged.array([[1, 2, 2], [3], [3, 3], [4, 4, 4], [4]])
    expected_unique_values = ragged.array([1, 2, 3, 4])
    expected_inverse_indices = ragged.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    unique_values, inverse_indices = ragged.unique_inverse(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(inverse_indices) == ak.to_list(expected_inverse_indices)


def test_can_take_scalar():
    arr = ragged.array([5])
    expected_unique_values = ragged.array([5])
    expected_unique_indices = ragged.array([0])
    unique_values, unique_indices = ragged.unique_inverse(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_indices) == ak.to_list(expected_unique_indices)


# unique_all tests
def test_can_all_none():
    with pytest.raises(TypeError):
        assert ragged.unique_all(ragged.array(None)) is None


def test_can_all_list():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "the truth value of an array whose length is not 1 is ambiguous;"
        ),
    ):
        assert ragged.unique_all(ragged.array([1, 2, 4, 3, 4, 5, 6, 20])) == (
            ragged.array([1, 2, 3, 4, 5, 6, 20]),
            ragged.array([0, 1, 3, 2, 5, 6, 7]),
            ragged.array([0, 1, 3, 2, 3, 4, 5, 6]),
            ragged.array([1, 1, 1, 2, 1, 1, 1]),
        )


def test_can_all_simple_array():
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


def test_can_all_scalar():
    arr = ragged.array([5])
    expected_unique_values = ragged.array([5])
    expected_unique_indices = ragged.array([0])
    expected_unique_inverse = ragged.array([0])
    expected_unique_counts = ragged.array([1])
    unique_values, unique_indices, unique_inverse, unique_counts = ragged.unique_all(
        arr
    )
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_indices) == ak.to_list(expected_unique_indices)
    assert ak.to_list(unique_inverse) == ak.to_list(expected_unique_inverse)
    assert ak.to_list(unique_counts) == ak.to_list(expected_unique_counts)
