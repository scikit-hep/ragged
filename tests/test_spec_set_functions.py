# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/set_functions.html
"""

from __future__ import annotations

import ragged

#Specific algorithm for unique_values:
#1 take an input array
#2 flatten input_array unless its 1d
#3 {remember the first element, loop through the rest of the list to see if there are copies
#    if yes then discard it and repeat the step
#    if not then add it to the output and repeat the step}
#4 once the cycle is over return an array of unique elements in the input array (the output must be of the same type as input array)

def test_existence():
    assert ragged.unique_all is not None
    assert ragged.unique_counts is not None
    assert ragged.unique_inverse is not None
    assert ragged.unique_values is not None
