# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/2022.12/API_specification/searching_functions.html
"""

from __future__ import annotations

from ..common._spec_searching_functions import argmax, argmin, nonzero, where

__all__ = [
    "argmax",
    "argmin",
    "nonzero",
    "where",
]
