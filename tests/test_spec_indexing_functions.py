# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/indexing_functions.html
"""

from __future__ import annotations

import numpy as np
import pytest

import ragged


def test_existence():
    assert ragged.take is not None


def test_take_1d():
    x = ragged.array([10, 20, 30, 40, 50])
    indices = ragged.array([0, 2, 4])
    result = ragged.take(x, indices)
    assert result.tolist() == [10, 30, 50]


def test_take_2d_axis0():
    x = ragged.array(np.arange(12).reshape(3, 4))
    indices = ragged.array([0, 2])
    result = ragged.take(x, indices, axis=0)
    assert result.tolist() == ragged.take(x, indices, axis=-2).tolist()
    assert result.tolist() == [[0, 1, 2, 3], [8, 9, 10, 11]]


def test_take_2d_axis1():
    x = ragged.array(np.arange(12).reshape(3, 4))
    indices = ragged.array([1, 3])
    result = ragged.take(x, indices, axis=1)
    assert result.tolist() == ragged.take(x, indices, axis=-1).tolist()
    assert result.tolist() == [[1, 3], [5, 7], [9, 11]]


def test_take_negative_axis_matches_positive():
    """Regression test: negative axis must wrap correctly (axis += ndim, not ndim+1)."""
    x = ragged.array(np.arange(6).reshape(2, 3))

    # axis=-2 must equal axis=0 on a 2-D array
    row_indices = ragged.array([0, 1])
    result_pos0 = ragged.take(x, row_indices, axis=0)
    result_neg2 = ragged.take(x, row_indices, axis=-2)
    assert result_pos0.tolist() == result_neg2.tolist()

    # axis=-1 must equal axis=1 on a 2-D array
    col_indices = ragged.array([0, 2])
    result_pos1 = ragged.take(x, col_indices, axis=1)
    result_neg1 = ragged.take(x, col_indices, axis=-1)
    assert result_pos1.tolist() == result_neg1.tolist()


def test_take_out_of_range_negative_axis():
    """Out-of-range negative axis must still raise AxisError."""
    import awkward as ak

    x = ragged.array(np.arange(6).reshape(2, 3))
    indices = ragged.array([0])
    with pytest.raises(ak.errors.AxisError, match="out of bounds"):
        ragged.take(x, indices, axis=-3)


def test_take_requires_axis_for_nd():
    x = ragged.array(np.arange(6).reshape(2, 3))
    indices = ragged.array([0])
    with pytest.raises(TypeError, match="axis"):
        ragged.take(x, indices)
