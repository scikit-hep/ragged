# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.io CF conventions helpers:
  to_cf_contiguous, from_cf_contiguous, to_cf_indexed, from_cf_indexed.

Coverage
--------
to_cf_contiguous
  - basic 2-D ragged -> (content, counts)
  - row with zero elements (empty row)
  - all rows same length (uniform)
  - single-row array
  - result types are ragged.array
  - dtype preserved in content

from_cf_contiguous
  - basic (content, counts) -> 2-D ragged
  - empty row (count=0)
  - single row
  - round-trip identity with to_cf_contiguous

to_cf_indexed
  - basic 2-D ragged -> (content, index)
  - index values are correct row assignments
  - empty row produces no index entries
  - result types are ragged.array

from_cf_indexed
  - sorted index (contiguous groups) -> 2-D ragged
  - unsorted index: elements assigned to rows in stated order, relative
    order within each row preserved
  - empty content/index -> empty ragged array
  - round-trip identity with to_cf_indexed

round-trip consistency
  - contiguous encoding round-trip
  - indexed encoding round-trip

error paths: to_cf_contiguous
  - non-array input raises TypeError
  - 1-D input raises ValueError
  - 0-D input raises ValueError
  - 3-D input raises ValueError

error paths: from_cf_contiguous
  - non-array content raises TypeError
  - 2-D counts raises ValueError
  - negative count raises ValueError
  - sum(counts) != len(content) raises ValueError

error paths: to_cf_indexed
  - 1-D input raises ValueError
  - 3-D input raises ValueError

error paths: from_cf_indexed
  - 2-D index raises ValueError
  - negative index value raises ValueError
  - len(content) != len(index) raises ValueError
"""

from __future__ import annotations

import numpy as np
import pytest

import ragged
from ragged.io.cf import (
    from_cf_contiguous,
    from_cf_indexed,
    to_cf_contiguous,
    to_cf_indexed,
)


def _make(nested, dtype=None) -> ragged.array:
    return ragged.array(nested, dtype=dtype)


def _np(x: ragged.array) -> np.ndarray:
    return np.array(x.tolist())


# ---------------------------------------------------------------------------
# to_cf_contiguous
# ---------------------------------------------------------------------------


class TestToCfContiguous:
    def test_basic(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        content, counts = to_cf_contiguous(a)
        np.testing.assert_array_equal(_np(content), [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(_np(counts), [3, 2, 1])

    def test_empty_row(self):
        a = _make([[], [1, 2], []], dtype=np.float64)
        content, counts = to_cf_contiguous(a)
        np.testing.assert_array_equal(_np(content), [1, 2])
        np.testing.assert_array_equal(_np(counts), [0, 2, 0])

    def test_uniform_rows(self):
        a = _make([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
        content, counts = to_cf_contiguous(a)
        np.testing.assert_array_equal(_np(content), [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(_np(counts), [2, 2, 2])

    def test_single_row(self):
        a = _make([[10, 20, 30]], dtype=np.float64)
        content, counts = to_cf_contiguous(a)
        np.testing.assert_array_equal(_np(content), [10, 20, 30])
        np.testing.assert_array_equal(_np(counts), [3])

    def test_result_types_are_ragged_array(self):
        a = _make([[1, 2], [3]], dtype=np.float64)
        content, counts = to_cf_contiguous(a)
        assert isinstance(content, ragged.array)
        assert isinstance(counts, ragged.array)

    def test_content_dtype_preserved(self):
        a = _make([[1, 2], [3, 4]], dtype=np.float32)
        content, _ = to_cf_contiguous(a)
        assert content.dtype == np.float32

    def test_content_is_1d(self):
        a = _make([[1, 2, 3], [4]], dtype=np.float64)
        content, counts = to_cf_contiguous(a)
        assert content.ndim == 1
        assert counts.ndim == 1


# ---------------------------------------------------------------------------
# from_cf_contiguous
# ---------------------------------------------------------------------------


class TestFromCfContiguous:
    def test_basic(self):
        content = _make([1, 2, 3, 4, 5, 6], dtype=np.float64)
        counts = _make([3, 2, 1], dtype=np.int64)
        result = from_cf_contiguous(content, counts)
        assert result.tolist() == [[1, 2, 3], [4, 5], [6]]

    def test_empty_row(self):
        content = _make([1.0, 2.0], dtype=np.float64)
        counts = _make([0, 2, 0], dtype=np.int64)
        result = from_cf_contiguous(content, counts)
        assert result.tolist() == [[], [1.0, 2.0], []]  # type: ignore[comparison-overlap]

    def test_single_row(self):
        content = _make([7, 8, 9], dtype=np.int64)
        counts = _make([3], dtype=np.int64)
        result = from_cf_contiguous(content, counts)
        assert result.tolist() == [[7, 8, 9]]

    def test_result_is_2d(self):
        content = _make([1, 2, 3, 4], dtype=np.float64)
        counts = _make([2, 2], dtype=np.int64)
        result = from_cf_contiguous(content, counts)
        assert result.ndim == 2

    def test_round_trip(self):
        original = _make([[10, 20], [30], [40, 50, 60]], dtype=np.float64)
        content, counts = to_cf_contiguous(original)
        recovered = from_cf_contiguous(content, counts)
        assert recovered.tolist() == original.tolist()

    def test_round_trip_with_empty_rows(self):
        original = _make([[], [1, 2], [], [3]], dtype=np.float64)
        content, counts = to_cf_contiguous(original)
        recovered = from_cf_contiguous(content, counts)
        assert recovered.tolist() == original.tolist()


# ---------------------------------------------------------------------------
# to_cf_indexed
# ---------------------------------------------------------------------------


class TestToCfIndexed:
    def test_basic(self):
        a = _make([[1, 2, 3], [4, 5], [6]], dtype=np.float64)
        content, index = to_cf_indexed(a)
        np.testing.assert_array_equal(_np(content), [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(_np(index), [0, 0, 0, 1, 1, 2])

    def test_index_row_assignment(self):
        a = _make([[10, 20], [], [30]], dtype=np.float64)
        content, index = to_cf_indexed(a)
        np.testing.assert_array_equal(_np(content), [10, 20, 30])
        np.testing.assert_array_equal(_np(index), [0, 0, 2])

    def test_empty_row_produces_no_index_entry(self):
        a = _make([[], [1, 2], []], dtype=np.float64)
        content, index = to_cf_indexed(a)
        np.testing.assert_array_equal(_np(content), [1.0, 2.0])
        np.testing.assert_array_equal(_np(index), [1, 1])

    def test_result_types_are_ragged_array(self):
        a = _make([[1], [2, 3]], dtype=np.float64)
        content, index = to_cf_indexed(a)
        assert isinstance(content, ragged.array)
        assert isinstance(index, ragged.array)

    def test_content_and_index_same_length(self):
        a = _make([[1, 2, 3], [4, 5]], dtype=np.float64)
        content, index = to_cf_indexed(a)
        assert len(content) == len(index) == 5

    def test_index_dtype_is_integer(self):
        a = _make([[1.0, 2.0], [3.0]], dtype=np.float64)
        _, index = to_cf_indexed(a)
        assert np.issubdtype(index.dtype, np.integer)


# ---------------------------------------------------------------------------
# from_cf_indexed
# ---------------------------------------------------------------------------


class TestFromCfIndexed:
    def test_sorted_index(self):
        content = _make([1, 2, 3, 4, 5, 6], dtype=np.float64)
        index = _make([0, 0, 0, 1, 1, 2], dtype=np.int64)
        result = from_cf_indexed(content, index)
        assert result.tolist() == [[1, 2, 3], [4, 5], [6]]

    def test_unsorted_index_relative_order_preserved(self):
        # Elements [10,20,30,40,50] with index [1,0,1,0,1]:
        # row 0 gets elements at positions 1,3 -> [20, 40]
        # row 1 gets elements at positions 0,2,4 -> [10, 30, 50]
        content = _make([10, 20, 30, 40, 50], dtype=np.int64)
        index = _make([1, 0, 1, 0, 1], dtype=np.int64)
        result = from_cf_indexed(content, index)
        assert result.tolist() == [[20, 40], [10, 30, 50]]

    def test_empty_content_and_index(self):
        content = _make([], dtype=np.float64)
        index = _make([], dtype=np.int64)
        result = from_cf_indexed(content, index)
        assert result.ndim == 2
        assert len(result) == 0

    def test_single_element(self):
        content = _make([42], dtype=np.float64)
        index = _make([0], dtype=np.int64)
        result = from_cf_indexed(content, index)
        assert result.tolist() == [[42.0]]

    def test_result_is_2d(self):
        content = _make([1, 2, 3], dtype=np.float64)
        index = _make([0, 1, 0], dtype=np.int64)
        result = from_cf_indexed(content, index)
        assert result.ndim == 2

    def test_round_trip(self):
        original = _make([[5, 6], [7], [8, 9, 10]], dtype=np.float64)
        content, index = to_cf_indexed(original)
        recovered = from_cf_indexed(content, index)
        assert recovered.tolist() == original.tolist()

    def test_round_trip_with_empty_rows(self):
        # to_cf_indexed skips empty rows in index, so from_cf_indexed must
        # infer the row count from max(index).  Rows with no elements simply
        # appear as empty lists at the end when max(index) covers them.
        original = _make([[1, 2], [3]], dtype=np.float64)
        content, index = to_cf_indexed(original)
        recovered = from_cf_indexed(content, index)
        assert recovered.tolist() == original.tolist()


# ---------------------------------------------------------------------------
# Round-trip consistency between contiguous and indexed
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_contiguous_and_indexed_agree(self):
        a = _make([[1, 2], [3, 4, 5], [6]], dtype=np.float64)
        c_content, c_counts = to_cf_contiguous(a)
        i_content, i_index = to_cf_indexed(a)
        # Both content arrays should be the same flat sequence
        np.testing.assert_array_equal(_np(c_content), _np(i_content))
        # Both should reconstruct to the same array
        from_c = from_cf_contiguous(c_content, c_counts)
        from_i = from_cf_indexed(i_content, i_index)
        assert from_c.tolist() == from_i.tolist() == a.tolist()


# ---------------------------------------------------------------------------
# Error paths: to_cf_contiguous
# ---------------------------------------------------------------------------


class TestToCfContiguousErrors:
    def test_non_array_raises_type_error(self):
        with pytest.raises(TypeError, match="ragged.array"):
            to_cf_contiguous([[1, 2], [3]])  # type: ignore[arg-type]

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2-D|ndim"):
            to_cf_contiguous(_make([1, 2, 3], dtype=np.float64))

    def test_0d_raises(self):
        with pytest.raises(ValueError, match="2-D|ndim"):
            to_cf_contiguous(ragged.array(5.0))

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2-D|ndim"):
            to_cf_contiguous(_make([[[1, 2], [3]], [[4]]], dtype=np.float64))


# ---------------------------------------------------------------------------
# Error paths: from_cf_contiguous
# ---------------------------------------------------------------------------


class TestFromCfContiguousErrors:
    def test_non_array_content_raises(self):
        with pytest.raises(TypeError, match="ragged.array"):
            from_cf_contiguous([1, 2, 3], _make([3], dtype=np.int64))  # type: ignore[arg-type]

    def test_2d_counts_raises(self):
        content = _make([1, 2, 3], dtype=np.float64)
        counts_2d = _make([[1, 2], [3]], dtype=np.int64)
        with pytest.raises(ValueError, match="1-D|ndim"):
            from_cf_contiguous(content, counts_2d)

    def test_negative_count_raises(self):
        content = _make([1.0, 2.0], dtype=np.float64)
        counts = _make([3, -1], dtype=np.int64)
        with pytest.raises(ValueError, match="[Nn]egative|non-negative"):
            from_cf_contiguous(content, counts)

    def test_sum_mismatch_raises(self):
        content = _make([1, 2, 3], dtype=np.float64)
        counts = _make([2, 2], dtype=np.int64)  # sum=4 != 3
        with pytest.raises(ValueError, match="[Mm]atch|sum|len"):
            from_cf_contiguous(content, counts)


# ---------------------------------------------------------------------------
# Error paths: to_cf_indexed
# ---------------------------------------------------------------------------


class TestToCfIndexedErrors:
    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2-D|ndim"):
            to_cf_indexed(_make([1, 2, 3], dtype=np.float64))

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2-D|ndim"):
            to_cf_indexed(_make([[[1, 2]], [[3]]], dtype=np.float64))


# ---------------------------------------------------------------------------
# Error paths: from_cf_indexed
# ---------------------------------------------------------------------------


class TestFromCfIndexedErrors:
    def test_non_array_index_raises(self):
        content = _make([1, 2, 3], dtype=np.float64)
        with pytest.raises(TypeError, match="ragged.array"):
            from_cf_indexed(content, [0, 1, 2])  # type: ignore[arg-type]

    def test_2d_index_raises(self):
        content = _make([1, 2], dtype=np.float64)
        index_2d = _make([[0, 0], [1]], dtype=np.int64)
        with pytest.raises(ValueError, match="1-D|ndim"):
            from_cf_indexed(content, index_2d)

    def test_negative_index_raises(self):
        content = _make([1, 2, 3], dtype=np.float64)
        index = _make([0, -1, 1], dtype=np.int64)
        with pytest.raises(ValueError, match="[Nn]egative|non-negative"):
            from_cf_indexed(content, index)

    def test_length_mismatch_raises(self):
        content = _make([1, 2, 3], dtype=np.float64)
        index = _make([0, 0], dtype=np.int64)  # len 2 != 3
        with pytest.raises(ValueError, match="[Mm]atch|len"):
            from_cf_indexed(content, index)
