# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Generic definitions used by the version-specific modules, such as
`ragged.v202212`.

https://data-apis.org/array-api/latest/API_specification/
"""

from __future__ import annotations

from ._creation import arange, asarray
from ._obj import array

__all__ = [
    # _creation
    "arange",
    "asarray",
    # _obj
    "array",
]
