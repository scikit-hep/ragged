import numpy as np
import awkward as ak
import ragged 

def unique_values1d(arr):
    if not isinstance(arr,ragged.array):
        print("input is not a ragged array")
    if arr.ndim != 1:
        print("input is not a 1D array")

    arr_list=ak.to_list(arr)
    arr_list.sort()
    
    seen=set()
    unique=[]
    for element in arr_list:
        if element not in seen:
            unique.append(element)
            seen.add(element)
    return ragged.array(unique)
