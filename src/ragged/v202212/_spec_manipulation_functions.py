# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/2022.12/API_specification/manipulation_functions.html
"""

from __future__ import annotations

from ..common._manipulation import (
    broadcast_arrays,
    broadcast_to,
    concat,
    expand_dims,
    flip,
    permute_dims,
    reshape,
    roll,
    squeeze,
    stack,
)

__all__ = [
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "permute_dims",
    "reshape",
    "roll",
    "squeeze",
    "stack",
]
