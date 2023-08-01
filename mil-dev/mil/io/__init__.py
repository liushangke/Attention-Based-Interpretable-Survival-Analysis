"""
This subpackage contains functions for reading and writing data
"""

# make functions available at the package level using shadow imports
from .reader import peek, read_record
from .transforms import flatten, parallel_dataset, structure
from .utils import study, inference
from .writer import convert_record, merge_records, split_inference, write_record

# list out things that are available for public use
__all__ = (
    # functions and classes of this package
    "convert_record",
    "flatten",
    "inference",
    "merge_records",
    "parallel_dataset",
    "peek",
    "read_record",
    "split_inference",
    "structure",
    "study",
    "write_record",
)
