# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.array.__setitem__ (issue #103 — array_api_extra.at support).

Coverage
--------
1-D uniform array
  - integer index set scalar
  - slice set scalar
  - slice set array
  - boolean mask set scalar
  - boolean mask set array
  - dtype preserved after mutation

2-D uniform array
  - integer row index set row
  - tuple index (row, col) set scalar
  - row slice set values
  - boolean mask on rows

2-D ragged array
  - integer row index: replace with shorter/longer/equal row
  - slice: replace multiple rows
  - unsupported key type raises TypeError

result invariants
  - shape updated correctly after mutation
  - dtype preserved
  - original impl not shared (mutations are isolated)

array_api_extra.at integration
  - at(x, idx).set(val) returns new array with correct values (copy semantics)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import ragged


def _make(data, dtype=None) -> ragged.array:
    return ragged.array(data, dtype=dtype)


# ---------------------------------------------------------------------------
# 1-D uniform
# ---------------------------------------------------------------------------


class TestSetitem1D:
    def test_integer_index(self):
        a = _make([1.0, 2.0, 3.0])
        a[1] = 99.0
        assert a.tolist() == [1.0, 99.0, 3.0]

    def test_slice_scalar(self):
        a = _make([1.0, 2.0, 3.0, 4.0])
        a[1:3] = 0.0
        assert a.tolist() == [1.0, 0.0, 0.0, 4.0]

    def test_slice_array(self):
        a = _make([1.0, 2.0, 3.0, 4.0])
        a[1:3] = _make([20.0, 30.0])
        assert a.tolist() == [1.0, 20.0, 30.0, 4.0]

    def test_boolean_mask_scalar(self):
        a = _make([1.0, 2.0, 3.0])
        mask = _make([True, False, True])
        a[mask] = 0.0
        assert a.tolist() == [0.0, 2.0, 0.0]

    def test_boolean_mask_array(self):
        a = _make([1.0, 2.0, 3.0])
        mask = _make([False, True, True])
        a[mask] = _make([20.0, 30.0])
        assert a.tolist() == [1.0, 20.0, 30.0]

    def test_dtype_preserved(self):
        a = _make([1.0, 2.0, 3.0], dtype=np.float32)
        a[0] = 99.0
        assert a.dtype == np.float32

    def test_shape_unchanged(self):
        a = _make([1.0, 2.0, 3.0])
        a[0] = 99.0
        assert a.shape == (3,)

    def test_negative_index(self):
        a = _make([1.0, 2.0, 3.0])
        a[-1] = 99.0
        assert a.tolist() == [1.0, 2.0, 99.0]


# ---------------------------------------------------------------------------
# 2-D uniform array
# ---------------------------------------------------------------------------


class TestSetitem2DUniform:
    def test_integer_row(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])
        a[0] = [10.0, 20.0]  # type: ignore[assignment]
        assert a.tolist() == [[10.0, 20.0], [3.0, 4.0]]

    def test_integer_row_ragged_array_value(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])
        a[1] = _make([30.0, 40.0])
        assert a.tolist() == [[1.0, 2.0], [30.0, 40.0]]

    def test_tuple_key(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])
        a[0, 1] = 99.0
        assert a.tolist() == [[1.0, 99.0], [3.0, 4.0]]

    def test_row_slice(self):
        a = _make([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        a[0:2] = [[10.0, 20.0], [30.0, 40.0]]  # type: ignore[assignment]
        assert a.tolist() == [[10.0, 20.0], [30.0, 40.0], [5.0, 6.0]]

    def test_boolean_mask_rows(self):
        a = _make([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = _make([True, False, True])
        a[mask] = [[0.0, 0.0], [0.0, 0.0]]  # type: ignore[assignment]
        assert a.tolist() == [[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]

    def test_dtype_preserved(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        a[0] = [10.0, 20.0]  # type: ignore[assignment]
        assert a.dtype == np.float32


# ---------------------------------------------------------------------------
# 2-D ragged array
# ---------------------------------------------------------------------------


class TestSetitem2DRagged:
    def test_integer_replace_same_length(self):
        a = _make([[1.0, 2.0, 3.0], [4.0, 5.0]])
        a[0] = [10.0, 20.0, 30.0]  # type: ignore[assignment]
        assert a.tolist() == [[10.0, 20.0, 30.0], [4.0, 5.0]]

    def test_integer_replace_different_length(self):
        a = _make([[1.0, 2.0, 3.0], [4.0, 5.0]])
        a[0] = [10.0, 20.0]  # type: ignore[assignment]  # shorter row
        assert a.tolist() == [[10.0, 20.0], [4.0, 5.0]]

    def test_integer_replace_longer(self):
        a = _make([[1.0, 2.0], [3.0]])
        a[1] = [30.0, 40.0, 50.0]  # type: ignore[assignment]  # longer row
        assert a.tolist() == [[1.0, 2.0], [30.0, 40.0, 50.0]]

    def test_slice_replace(self):
        a = _make([[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]])
        a[0:2] = [[10.0], [20.0, 30.0, 40.0, 50.0]]  # type: ignore[assignment]
        assert a.tolist() == [[10.0], [20.0, 30.0, 40.0, 50.0], [6.0]]

    def test_negative_index(self):
        a = _make([[1.0, 2.0], [3.0, 4.0, 5.0]])
        a[-1] = [30.0]  # type: ignore[assignment]
        assert a.tolist() == [[1.0, 2.0], [30.0]]

    def test_dtype_preserved(self):
        a = _make([[1.0, 2.0], [3.0]], dtype=np.float64)
        a[0] = [10.0, 20.0]  # type: ignore[assignment]
        assert a.dtype == np.float64

    def test_unsupported_key_raises(self):
        a = _make([[1.0, 2.0], [3.0]])
        mask = _make([True, False])
        with pytest.raises(TypeError, match="integer and slice keys"):
            a[mask] = [[0.0, 0.0]]  # type: ignore[assignment]

    def test_value_as_ragged_array(self):
        a = _make([[1.0, 2.0], [3.0, 4.0, 5.0]])
        a[0] = _make([10.0, 20.0])
        assert a.tolist() == [[10.0, 20.0], [3.0, 4.0, 5.0]]


# ---------------------------------------------------------------------------
# Isolation: __setitem__ on a copy does not affect the original
# ---------------------------------------------------------------------------


class TestSetitemIsolation:
    def test_copy_is_isolated_1d(self):
        import copy

        a = _make([1.0, 2.0, 3.0])
        b = copy.copy(a)
        b[0] = 99.0
        assert a.tolist() == [1.0, 2.0, 3.0]
        assert b.tolist() == [99.0, 2.0, 3.0]

    def test_copy_is_isolated_ragged(self):
        import copy

        a = _make([[1.0, 2.0], [3.0]])
        b = copy.copy(a)
        b[0] = [10.0, 20.0]  # type: ignore[assignment]
        assert a.tolist() == [[1.0, 2.0], [3.0]]
        assert b.tolist() == [[10.0, 20.0], [3.0]]


# ---------------------------------------------------------------------------
# Simulate array_api_extra.at usage (copy + setitem)
# ---------------------------------------------------------------------------


class TestAtSimulation:
    """Simulate what array_api_extra.at does: copy then __setitem__."""

    def _at_set(self, x: ragged.array, idx: Any, val: Any) -> ragged.array:
        import copy

        out = copy.copy(x)
        out[idx] = val
        return out

    def test_1d_set(self):
        a = _make([1.0, 2.0, 3.0])
        b = self._at_set(a, 1, 99.0)
        assert b.tolist() == [1.0, 99.0, 3.0]
        assert a.tolist() == [1.0, 2.0, 3.0]  # original unchanged

    def test_2d_uniform_set_row(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])
        b = self._at_set(a, 0, [10.0, 20.0])
        assert b.tolist() == [[10.0, 20.0], [3.0, 4.0]]

    def test_2d_ragged_set_row(self):
        a = _make([[1.0, 2.0, 3.0], [4.0]])
        b = self._at_set(a, 1, [40.0, 50.0])
        assert b.tolist() == [[1.0, 2.0, 3.0], [40.0, 50.0]]
