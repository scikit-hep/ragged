# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Generic definitions used by the version-specific modules, such as
`ragged.v202212`.

https://data-apis.org/array-api/latest/API_specification/
"""

from __future__ import annotations

from ._creation import (
    arange,
    asarray,
    empty,
    empty_like,
    eye,
    from_dlpack,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)
from ._datatype import (
    astype,
)
from ._obj import array

__all__ = [
    # _creation
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
    # _datatype
    "astype",
    # _obj
    "array",
]
