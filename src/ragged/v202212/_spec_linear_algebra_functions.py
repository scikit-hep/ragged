# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/2022.12/API_specification/linear_algebra_functions.html
"""

from __future__ import annotations

from ..common._spec_linear_algebra_functions import (
    matmul,
    matrix_transpose,
    tensordot,
    vecdot,
)

__all__ = ["matmul", "matrix_transpose", "tensordot", "vecdot"]
