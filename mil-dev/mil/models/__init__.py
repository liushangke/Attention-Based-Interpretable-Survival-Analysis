"""
This subpackage contains functions for creating models
"""

# make functions available at the package level using shadow imports
from .convolutional import convolutional_model
from .dense import attention_flat, attention_flat_tune, attention_flat_train
from .utils import isragged

# list out things that are available for public use
__all__ = (
    # functions and classes of this package
    "attention_flat",
    "convolutional_model",
    "isragged",
    "attention_flat_tune",
    "attention_flat_train",
)
