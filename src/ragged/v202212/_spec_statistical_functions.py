# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/2022.12/API_specification/statistical_functions.html
"""

from __future__ import annotations

from ..common._statistical import (  # pylint: disable=W0622
    max,
    mean,
    min,
    prod,
    std,
    sum,
    var,
)

__all__ = ["max", "mean", "min", "prod", "std", "sum", "var"]
