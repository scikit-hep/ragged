# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/2022.12/API_specification/data_type_functions.html
"""

from __future__ import annotations

from ..common._datatype import (
    astype,
    can_cast,
    finfo,
    iinfo,
    isdtype,
    result_type,
)

__all__ = [
    "astype",
    "can_cast",
    "finfo",
    "iinfo",
    "isdtype",
    "result_type",
]
