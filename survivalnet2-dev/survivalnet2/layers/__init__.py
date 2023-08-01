"""
This package contains layers for operations like clustering and gradient reversal
"""
# make functions available at the package level using shadow imports
from .gradient_reversal import GradientReversal
from .advmtl import advmtl_model, advmtl_space, optimization_space, task_space

# list out things that are available for public use
__all__ = (
    # functions and classes of this package
    "advmtl_model",
    "advmtl_space",
    "GradientReversal",
    "optimization_space",
    "task_space",
)
