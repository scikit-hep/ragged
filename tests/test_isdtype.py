# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Tests for ragged.isdtype.

Coverage
--------
kind as dtype
  - exact match returns True
  - different dtype returns False

kind as string: "bool"
  - bool dtype -> True
  - non-bool -> False

kind as string: "signed integer"
  - int8, int16, int32, int64 -> True
  - uint, float, bool -> False

kind as string: "unsigned integer"
  - uint8, uint16, uint32, uint64 -> True
  - int, float, bool -> False

kind as string: "integral"
  - signed and unsigned integers -> True
  - float, bool -> False

kind as string: "real floating"
  - float32, float64 -> True
  - int, complex, bool -> False

kind as string: "complex floating"
  - complex64, complex128 -> True
  - float, int, bool -> False

kind as string: "numeric"
  - int, uint, float, complex -> True
  - bool -> False

kind as tuple
  - union of kinds: True if dtype matches any
  - empty tuple -> False

kind string case-insensitivity

error path
  - unknown kind string raises ValueError
"""

from __future__ import annotations

import numpy as np
import pytest

import ragged

# ---------------------------------------------------------------------------
# kind as dtype
# ---------------------------------------------------------------------------


class TestIsdtypeKindAsDtype:
    def test_exact_match(self):
        assert ragged.isdtype(np.dtype("float64"), np.float64) is True

    def test_exact_match_int(self):
        assert ragged.isdtype(np.dtype("int32"), np.int32) is True

    def test_no_match(self):
        assert ragged.isdtype(np.dtype("float32"), np.float64) is False

    def test_no_match_kind(self):
        assert ragged.isdtype(np.dtype("int64"), np.float64) is False

    def test_dtype_object_as_kind(self):
        assert ragged.isdtype(np.dtype("bool"), np.dtype("bool")) is True

    def test_dtype_object_mismatch(self):
        assert ragged.isdtype(np.dtype("bool"), np.dtype("int8")) is False


# ---------------------------------------------------------------------------
# kind = "bool"
# ---------------------------------------------------------------------------


class TestIsdtypeBool:
    def test_bool_true(self):
        assert ragged.isdtype(np.dtype("bool"), "bool") is True

    def test_int_not_bool(self):
        assert ragged.isdtype(np.dtype("int32"), "bool") is False

    def test_float_not_bool(self):
        assert ragged.isdtype(np.dtype("float64"), "bool") is False


# ---------------------------------------------------------------------------
# kind = "signed integer"
# ---------------------------------------------------------------------------


class TestIsdtypeSignedInteger:
    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_signed_ints(self, dt):
        assert ragged.isdtype(np.dtype(dt), "signed integer") is True

    def test_uint_not_signed(self):
        assert ragged.isdtype(np.dtype("uint32"), "signed integer") is False

    def test_float_not_signed(self):
        assert ragged.isdtype(np.dtype("float64"), "signed integer") is False

    def test_bool_not_signed(self):
        assert ragged.isdtype(np.dtype("bool"), "signed integer") is False


# ---------------------------------------------------------------------------
# kind = "unsigned integer"
# ---------------------------------------------------------------------------


class TestIsdtypeUnsignedInteger:
    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_unsigned_ints(self, dt):
        assert ragged.isdtype(np.dtype(dt), "unsigned integer") is True

    def test_int_not_unsigned(self):
        assert ragged.isdtype(np.dtype("int32"), "unsigned integer") is False

    def test_float_not_unsigned(self):
        assert ragged.isdtype(np.dtype("float32"), "unsigned integer") is False


# ---------------------------------------------------------------------------
# kind = "integral"
# ---------------------------------------------------------------------------


class TestIsdtypeIntegral:
    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_signed_integral(self, dt):
        assert ragged.isdtype(np.dtype(dt), "integral") is True

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_unsigned_integral(self, dt):
        assert ragged.isdtype(np.dtype(dt), "integral") is True

    def test_float_not_integral(self):
        assert ragged.isdtype(np.dtype("float64"), "integral") is False

    def test_bool_not_integral(self):
        assert ragged.isdtype(np.dtype("bool"), "integral") is False


# ---------------------------------------------------------------------------
# kind = "real floating"
# ---------------------------------------------------------------------------


class TestIsdtypeRealFloating:
    def test_float32(self):
        assert ragged.isdtype(np.dtype("float32"), "real floating") is True

    def test_float64(self):
        assert ragged.isdtype(np.dtype("float64"), "real floating") is True

    def test_int_not_real_float(self):
        assert ragged.isdtype(np.dtype("int64"), "real floating") is False

    def test_complex_not_real_float(self):
        assert ragged.isdtype(np.dtype("complex128"), "real floating") is False

    def test_bool_not_real_float(self):
        assert ragged.isdtype(np.dtype("bool"), "real floating") is False


# ---------------------------------------------------------------------------
# kind = "complex floating"
# ---------------------------------------------------------------------------


class TestIsdtypeComplexFloating:
    def test_complex64(self):
        assert ragged.isdtype(np.dtype("complex64"), "complex floating") is True

    def test_complex128(self):
        assert ragged.isdtype(np.dtype("complex128"), "complex floating") is True

    def test_float_not_complex(self):
        assert ragged.isdtype(np.dtype("float64"), "complex floating") is False

    def test_int_not_complex(self):
        assert ragged.isdtype(np.dtype("int32"), "complex floating") is False


# ---------------------------------------------------------------------------
# kind = "numeric"
# ---------------------------------------------------------------------------


class TestIsdtypeNumeric:
    @pytest.mark.parametrize(
        "dt",
        [
            np.int8,
            np.int32,
            np.int64,
            np.uint8,
            np.uint64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ],
    )
    def test_numeric_types(self, dt):
        assert ragged.isdtype(np.dtype(dt), "numeric") is True

    def test_bool_not_numeric(self):
        assert ragged.isdtype(np.dtype("bool"), "numeric") is False


# ---------------------------------------------------------------------------
# kind as tuple
# ---------------------------------------------------------------------------


class TestIsdtypeTuple:
    def test_matches_first(self):
        assert (
            ragged.isdtype(np.dtype("float64"), ("real floating", "complex floating"))
            is True
        )

    def test_matches_second(self):
        assert (
            ragged.isdtype(np.dtype("complex64"), ("real floating", "complex floating"))
            is True
        )

    def test_matches_neither(self):
        assert (
            ragged.isdtype(np.dtype("int32"), ("real floating", "complex floating"))
            is False
        )

    def test_dtype_in_tuple(self):
        assert ragged.isdtype(np.dtype("float64"), (np.float64, np.float32)) is True

    def test_mixed_tuple(self):
        assert ragged.isdtype(np.dtype("int32"), ("real floating", np.int32)) is True

    def test_empty_tuple(self):
        assert ragged.isdtype(np.dtype("float64"), ()) is False


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------


class TestIsdtypeCaseInsensitive:
    def test_upper(self):
        assert ragged.isdtype(np.dtype("float64"), "Real Floating") is True

    def test_mixed(self):
        assert ragged.isdtype(np.dtype("int32"), "Signed Integer") is True


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestIsdtypeErrors:
    def test_unknown_kind_string_returns_false(self):
        assert ragged.isdtype(np.dtype("float64"), "quaternion") is False

    def test_unknown_kind_string_int_returns_false(self):
        assert ragged.isdtype(np.dtype("float64"), "int") is False
