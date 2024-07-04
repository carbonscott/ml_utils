import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import torch
import numpy as np

import numpy as np
from typing import List


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return None


def print_layers(module, max_depth=1, current_indent_width=0, prints_module_name=True):
    """
    Recursively prints the layers of a PyTorch module.  (Keep printing child
    element with a depth first search approach.)

    Args:
    - module (nn.Module): The current module or layer to print.
    - current_indent_width (int): The current level of indentation for printing.
    - prints_name (bool): Flag to determine if the name of the module should be printed.
    """

    def _print_current_layer(module, depth=0, current_indent_width=0, prints_module_name=True):
        # Define a prefix based on current indent level
        prefix = '  ' * current_indent_width

        # Print the name and type of the current module
        if prints_module_name: print(f"{module.__class__.__name__}", end = "")
        print()

        # Check if the current module has children
        # If it does, recursively print each child with an increased indentation level
        if depth < max_depth and list(module.children()):
            for name, child in module.named_children():
                print(f"{prefix}- ({name}): ", end = "")
                _print_current_layer(child, depth + 1, current_indent_width + 1, prints_module_name)

    _print_current_layer(module, current_indent_width=current_indent_width, prints_module_name=prints_module_name)


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def find_power_of_2_friendly(start_num: int, power_range: List[int], scan_range: List[int]):
    """
    Find the nearest number to start_num that is divisible by the most powers of 2.

    Args:
    start_num (int): The starting number to scan from.
    power_range (List[int]): The powers of 2 to check (e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).
    scan_range (List[int]): The range of numbers to scan (e.g., range(100) to check the next 100 numbers).

    Returns:
    tuple: (friendly_number, offset, divisibility_count)
        friendly_number: The found power of 2 friendly number.
        offset: The difference between friendly_number and start_num.
        divisibility_count: The number of powers of 2 that divide the friendly_number.

    Example:
    ```
    start_num = 50257
    power_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2^0 to 2^9
    scan_range = range(100)  # Check the next 100 numbers

    result = find_power_of_2_friendly(start_num, power_range, scan_range)
    print(f"Friendly number: {result[0]}")
    print(f"Offset from start: {result[1]}")
    print(f"Divisible by {result[2]} powers of 2")

    # Additional examples
    print("\nUsing lists:")
    result = find_power_of_2_friendly(50257, [1, 2, 4, 8], [0, 1, 2, 3, 4, 5])
    print(f"Result: {result}")

    print("\nUsing NumPy arrays:")
    result = find_power_of_2_friendly(50257, np.arange(5), np.arange(10))
    print(f"Result: {result}")
    ```
    """
    # Create a 2D boolean array of divisibility checks
    divisibility_matrix = np.array([
        [(start_num + n) % (2**i) == 0 for i in power_range]
        for n in scan_range
    ])

    # Sum along the power axis to get divisibility counts
    divisibility_counts = divisibility_matrix.sum(axis=1)

    # Find the index of the maximum divisibility count
    best_offset_index = divisibility_counts.argmax()

    # Calculate the friendly number and its divisibility count
    best_offset = scan_range[best_offset_index]
    friendly_number = start_num + best_offset
    max_divisibility_count = divisibility_counts[best_offset_index]

    return friendly_number, best_offset, max_divisibility_count
