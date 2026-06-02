# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Regression tests for ragged.array.__iter__ shape correctness (#96).

The bug: iterating over an N-D array yielded subarrays whose shape was
(len(x), d2, d3, ...) instead of the correct (len(x), d3, ...).  In other
words ``len(x)`` was prepended to the already-sliced ``shape[1:]``, making
every yielded element appear one dimension too large.

Coverage
--------
2-D uniform array
  - yielded elements have ndim=1 and correct shape
  - repr works without error
  - values match direct indexing

2-D ragged array
  - yielded elements have ndim=1 with per-row length
  - shape equals (row_len,)

3-D uniform array
  - yielded elements have ndim=2 and shape (d1, None)

3-D ragged array
  - yielded elements have ndim=2

1-D array
  - yields 0-D scalar wrappers (existing behaviour unchanged)
  - iteration over 0-D raises TypeError
"""

from __future__ import annotations

import numpy as np
import pytest

import ragged

# ---------------------------------------------------------------------------
# 2-D uniform array (the primary bug case from the issue)
# ---------------------------------------------------------------------------


class TestIter2DUniform:
    def test_element_ndim(self):
        A = ragged.array(np.arange(100).reshape(10, 10))
        for row in A:
            assert row.ndim == 1

    def test_element_shape(self):
        A = ragged.array(np.arange(100).reshape(10, 10))
        rows = list(A)
        for row in rows:
            assert row.shape == (10,)

    def test_element_shape_rectangular(self):
        A = ragged.array(np.arange(12).reshape(3, 4), dtype=np.float64)
        for row in A:
            assert row.shape == (4,)

    def test_element_values_match_indexing(self):
        A = ragged.array(np.arange(100).reshape(10, 10))
        rows = list(A)
        for i, row in enumerate(rows):
            np.testing.assert_array_equal(row.tolist(), A[i].tolist())

    def test_repr_does_not_raise(self):
        """repr crashed before the fix (AssertionError in ak.prettyprint)."""
        A = ragged.array(np.arange(100).reshape(10, 10))
        rows = list(A)
        for row in rows:
            _ = repr(row)  # must not raise

    def test_ak_to_numpy_works(self):
        """ak.to_numpy(row) must work; uniform iter yields numpy arrays as _impl."""
        import awkward as ak

        A = ragged.array(np.arange(100).reshape(10, 10))
        rows = list(A)
        for i, row in enumerate(rows):
            np.testing.assert_array_equal(
                ak.to_numpy(row), np.arange(i * 10, (i + 1) * 10)
            )

    def test_next_iter_shape(self):
        A = ragged.array(np.arange(100).reshape(10, 10))
        first = next(iter(A))
        assert first.shape == (10,)

    def test_dtype_preserved(self):
        A = ragged.array(np.arange(6, dtype=np.float32).reshape(2, 3))
        for row in A:
            assert row.dtype == np.float32


# ---------------------------------------------------------------------------
# 2-D ragged array
# ---------------------------------------------------------------------------


class TestIter2DRagged:
    def test_element_ndim(self):
        R = ragged.array([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        for row in R:
            assert row.ndim == 1

    def test_element_shapes(self):
        R = ragged.array([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        shapes = [row.shape for row in R]
        assert shapes == [(3,), (2,), (1,)]

    def test_element_values(self):
        R = ragged.array([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        rows = list(R)
        assert rows[0].tolist() == [1.0, 2.0, 3.0]
        assert rows[1].tolist() == [4.0, 5.0]
        assert rows[2].tolist() == [6.0]

    def test_empty_row(self):
        R = ragged.array([[], [1, 2]], dtype=np.float64)
        rows = list(R)
        assert rows[0].shape == (0,)
        assert rows[1].shape == (2,)

    def test_dtype_preserved(self):
        R = ragged.array([[1, 2], [3]], dtype=np.int64)
        for row in R:
            assert row.dtype == np.int64


# ---------------------------------------------------------------------------
# 3-D uniform array
# ---------------------------------------------------------------------------


class TestIter3DUniform:
    def test_element_ndim(self):
        T = ragged.array(np.arange(24).reshape(2, 3, 4).tolist(), dtype=np.float64)
        for slab in T:
            assert slab.ndim == 2

    def test_element_shape(self):
        T = ragged.array(np.arange(24).reshape(2, 3, 4).tolist(), dtype=np.float64)
        for slab in T:
            assert slab.shape == (3, None)

    def test_element_values(self):
        a_np = np.arange(24).reshape(2, 3, 4)
        T = ragged.array(a_np.tolist(), dtype=np.float64)
        slabs = list(T)
        for i, slab in enumerate(slabs):
            np.testing.assert_array_equal(np.array(slab.tolist()), a_np[i])


# ---------------------------------------------------------------------------
# 3-D ragged array
# ---------------------------------------------------------------------------


class TestIter3DRagged:
    def test_element_ndim(self):
        R3 = ragged.array([[[1, 2], [3]], [[4], [5, 6, 7]]], dtype=np.float64)
        for outer in R3:
            assert outer.ndim == 2

    def test_element_shape(self):
        R3 = ragged.array([[[1, 2], [3]], [[4], [5, 6, 7]]], dtype=np.float64)
        outers = list(R3)
        assert outers[0].shape == (2, None)
        assert outers[1].shape == (2, None)


# ---------------------------------------------------------------------------
# 1-D array and 0-D error
# ---------------------------------------------------------------------------


class TestIter1D:
    def test_yields_scalars(self):
        a = ragged.array([10, 20, 30], dtype=np.int64)
        vals = [x.tolist() for x in a]
        assert vals == [10, 20, 30]

    def test_zero_d_raises(self):
        a = ragged.array([42], dtype=np.int64)
        scalar = a[0]
        with pytest.raises(TypeError, match="0-d"):
            list(scalar)
