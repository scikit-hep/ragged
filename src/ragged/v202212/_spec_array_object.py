# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/2022.12/API_specification/array_object.html
"""

from __future__ import annotations

from ..common._obj import array as common_array


class array(common_array):  # pylint: disable=C0103
    """
    Ragged array class and constructor for data-apis.org/array-api/2022.12.
    """


__all__ = ["array"]
