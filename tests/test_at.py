# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.array.at (JAX-style functional updates, issue #103).

Coverage
--------
.at[idx].set(val)
  - integer index, scalar value
  - slice index, scalar value
  - slice index, array value
  - boolean mask
  - original array is unchanged (copy semantics)

.at[idx].add(val)
.at[idx].subtract(val)
.at[idx].multiply(val)
.at[idx].divide(val)
.at[idx].power(val)
  - 1-D integer index
  - 1-D slice

.at[idx].min(val)
.at[idx].max(val)
  - clamp element below / above threshold

2-D uniform arrays
2-D ragged arrays
  - integer row index
  - slice

dtype preserved throughout
result type is always ragged.array
"""

from __future__ import annotations

import numpy as np

import ragged


def _make(data, dtype=None) -> ragged.array:
    return ragged.array(data, dtype=dtype)


# ---------------------------------------------------------------------------
# .at[idx].set
# ---------------------------------------------------------------------------


class TestAtSet:
    def test_scalar_int_index(self):
        a = _make([1.0, 2.0, 3.0])
        b = a.at[1].set(99.0)
        assert b.tolist() == [1.0, 99.0, 3.0]

    def test_original_unchanged(self):
        a = _make([1.0, 2.0, 3.0])
        _ = a.at[0].set(99.0)
        assert a.tolist() == [1.0, 2.0, 3.0]

    def test_negative_index(self):
        a = _make([1.0, 2.0, 3.0])
        assert a.at[-1].set(0.0).tolist() == [1.0, 2.0, 0.0]

    def test_slice_scalar(self):
        a = _make([1.0, 2.0, 3.0, 4.0])
        assert a.at[1:3].set(0.0).tolist() == [1.0, 0.0, 0.0, 4.0]

    def test_slice_array(self):
        a = _make([1.0, 2.0, 3.0, 4.0])
        assert a.at[1:3].set(_make([20.0, 30.0])).tolist() == [1.0, 20.0, 30.0, 4.0]

    def test_boolean_mask(self):
        a = _make([1.0, 2.0, 3.0])
        mask = _make([True, False, True])
        assert a.at[mask].set(0.0).tolist() == [0.0, 2.0, 0.0]

    def test_dtype_preserved(self):
        a = _make([1.0, 2.0, 3.0], dtype=np.float32)
        b = a.at[0].set(99.0)
        assert b.dtype == np.float32

    def test_result_type(self):
        a = _make([1.0, 2.0])
        assert isinstance(a.at[0].set(0.0), ragged.array)

    def test_2d_uniform_row(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])
        b = a.at[0].set(_make([10.0, 20.0]))
        assert b.tolist() == [[10.0, 20.0], [3.0, 4.0]]

    def test_2d_uniform_element(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])
        b = a.at[0, 1].set(99.0)
        assert b.tolist() == [[1.0, 99.0], [3.0, 4.0]]

    def test_2d_ragged_row(self):
        a = _make([[1.0, 2.0, 3.0], [4.0, 5.0]])
        b = a.at[1].set(_make([40.0, 50.0]))
        assert b.tolist() == [[1.0, 2.0, 3.0], [40.0, 50.0]]

    def test_2d_ragged_slice(self):
        a = _make([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]])
        b = a.at[0:2].set([[10.0, 20.0], [30.0]])
        assert b.tolist() == [[10.0, 20.0], [30.0], [4.0, 5.0, 6.0]]


# ---------------------------------------------------------------------------
# .at[idx].add / subtract / multiply / divide / power
# ---------------------------------------------------------------------------


class TestAtArithmetic:
    def test_add_scalar(self):
        a = _make([1.0, 2.0, 3.0])
        assert a.at[0].add(10.0).tolist() == [11.0, 2.0, 3.0]

    def test_add_slice(self):
        a = _make([1.0, 2.0, 3.0])
        assert a.at[1:3].add(10.0).tolist() == [1.0, 12.0, 13.0]

    def test_subtract(self):
        a = _make([1.0, 2.0, 3.0])
        assert a.at[2].subtract(0.5).tolist() == [1.0, 2.0, 2.5]

    def test_multiply(self):
        a = _make([1.0, 2.0, 3.0])
        assert a.at[1:3].multiply(2.0).tolist() == [1.0, 4.0, 6.0]

    def test_divide(self):
        a = _make([1.0, 2.0, 4.0])
        assert a.at[2].divide(2.0).tolist() == [1.0, 2.0, 2.0]

    def test_power(self):
        a = _make([1.0, 2.0, 3.0])
        assert a.at[2].power(2.0).tolist() == [1.0, 2.0, 9.0]

    def test_original_unchanged_after_add(self):
        a = _make([1.0, 2.0, 3.0])
        _ = a.at[0].add(100.0)
        assert a.tolist() == [1.0, 2.0, 3.0]

    def test_dtype_preserved_after_add(self):
        a = _make([1.0, 2.0, 3.0], dtype=np.float32)
        b = a.at[0].add(1.0)
        assert b.dtype == np.float32

    def test_add_2d_uniform(self):
        a = _make([[1.0, 2.0], [3.0, 4.0]])
        b = a.at[0].add(_make([10.0, 20.0]))
        assert b.tolist() == [[11.0, 22.0], [3.0, 4.0]]

    def test_add_2d_ragged(self):
        a = _make([[1.0, 2.0, 3.0], [4.0, 5.0]])
        b = a.at[0].add(_make([10.0, 20.0, 30.0]))
        assert b.tolist() == [[11.0, 22.0, 33.0], [4.0, 5.0]]


# ---------------------------------------------------------------------------
# .at[idx].min / .max  (clamp operations)
# ---------------------------------------------------------------------------


class TestAtMinMax:
    def test_min_clamps_above(self):
        a = _make([1.0, 5.0, 3.0])
        # at[1].min(2.0) → x[1] = min(5.0, 2.0) = 2.0
        b = a.at[1].min(2.0)
        assert b.tolist() == [1.0, 2.0, 3.0]

    def test_min_no_change_when_below(self):
        a = _make([1.0, 5.0, 3.0])
        b = a.at[1].min(10.0)  # min(5.0, 10.0) = 5.0 → unchanged
        assert b.tolist() == [1.0, 5.0, 3.0]

    def test_max_clamps_below(self):
        a = _make([1.0, 0.5, 3.0])
        # at[1].max(2.0) → x[1] = max(0.5, 2.0) = 2.0
        b = a.at[1].max(2.0)
        assert b.tolist() == [1.0, 2.0, 3.0]

    def test_max_no_change_when_above(self):
        a = _make([1.0, 5.0, 3.0])
        b = a.at[1].max(0.0)  # max(5.0, 0.0) = 5.0 → unchanged
        assert b.tolist() == [1.0, 5.0, 3.0]

    def test_original_unchanged(self):
        a = _make([1.0, 5.0, 3.0])
        _ = a.at[1].min(2.0)
        assert a.tolist() == [1.0, 5.0, 3.0]
