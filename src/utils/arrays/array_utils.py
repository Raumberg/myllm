import numpy as np
from typing import List

def filter_indices(a: List[int], b: List[int]) -> List[int]:
    """
    Filters elements from list `b` based on the values in list `a`.

    For each element in `a`, this function finds the next element in `b` 
    that is greater than the current element in `a` and appends it to 
    the result list. The function ensures that each element from `b` 
    is only used once.

    Args:
        a (List[int]): A list of integers to filter against.
        b (List[int]): A list of integers from which to filter elements.

    Returns:
        List[int]: A list of filtered elements from `b` that are greater 
                    than the corresponding elements in `a`.
    """
    filtered_b = []
    a_len = len(a)
    b_len = len(b)

    j = 0  # Pointer for list b

    for i in range(a_len):
        while j < b_len and b[j] <= a[i]:
            j += 1
        if j < b_len:
            filtered_b.append(b[j])
            j += 1  

    return filtered_b

def find_occurrences_v3(arr: List[int], subarr: List[int]) -> List[int]:
    """
    Finds the starting indices of occurrences of a subarray within an array.

    This function uses convolution to identify positions in `arr` where 
    the sum of a sliding window matches the sum of `subarr`. It returns 
    the starting indices of these occurrences.

    Args:
        arr (List[int]): The main array in which to search for occurrences.
        subarr (List[int]): The subarray to find within `arr`.

    Returns:
        List[int]: A list of starting indices where `subarr` occurs in `arr`.
    """
    m = len(subarr)
    conv_result = np.convolve(arr, subarr[::-1], mode='valid')
    subarr_sum = np.sum(subarr)
    window_sum = np.convolve(arr, np.ones(m), mode='valid')
    positions = np.where((conv_result == subarr_sum * window_sum) & (window_sum == subarr_sum))[0]
    return positions.tolist()