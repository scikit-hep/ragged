# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/version.html
"""

from __future__ import annotations

import ragged


def test_values():
    assert ragged.v202212.__array_api_version__ == "2022.12"

    assert ragged.__array_api_version__ == "2022.12"
