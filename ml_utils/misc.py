import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import torch
import numpy as np


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
