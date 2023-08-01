"""
This subpackage contains functions for manipulating and normalizing data
"""

# make functions available at the package level using shadow imports
from .labels import stack_labels
from .labels import unstack_labels

# list out things that are available for public use
__all__ = (
    # functions and classes of this package
    "stack_labels",
    "unstack_labels",
)
