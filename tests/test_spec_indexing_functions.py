# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/indexing_functions.html
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

import ragged


def _make(data, dtype=None) -> ragged.array:
    return ragged.array(data, dtype=dtype)


def test_existence():
    assert ragged.take is not None
    assert ragged.take_along_axis is not None


class TestTakeAlongAxis:
    def test_1d_uniform(self):
        a = _make([10, 20, 30, 40], dtype=np.int64)
        indices = _make([2, 0, 3], dtype=np.int64)
        result = ragged.take_along_axis(a, indices, axis=0)
        assert ak.to_list(result) == [30, 10, 40]
        assert result.dtype == np.int64

    def test_2d_uniform_numpy_fast_path(self):
        a = _make([[10, 20, 30], [40, 50, 60]])
        indices = _make([[0, 2], [1, 1]], dtype=np.int64)
        result = ragged.take_along_axis(a, indices, axis=1)
        assert ak.to_list(result) == [[10, 30], [50, 50]]

    def test_2d_ragged_innermost_axis(self):
        # Ragged array innermost dimension gather
        a = _make([[10, 20, 30], [40, 50]])
        indices = _make([[2, 0], [1]], dtype=np.int64)
        result = ragged.take_along_axis(a, indices, axis=1)
        assert ak.to_list(result) == [[30, 10], [50]]

    def test_error_rank_mismatch(self):
        a = _make([[10, 20], [30, 40]])
        indices = _make([0, 1], dtype=np.int64)
        with pytest.raises(ValueError, match="same rank"):
            ragged.take_along_axis(a, indices, axis=0)

    def test_error_axis_out_of_bounds(self):
        a = _make([10, 20])
        indices = _make([0])
        with pytest.raises(ValueError, match="out of bounds"):
            ragged.take_along_axis(a, indices, axis=1)
