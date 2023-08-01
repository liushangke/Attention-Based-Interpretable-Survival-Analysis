"""
This package contains tensorflow losses for optimizing time-to-event models
"""

# make functions available at the package level using shadow imports
from .cox import cox
from .cox import efron
from .parametric import Exponential, Weibull, Gompertz


# define number of inputs and input domain constraints for losses
LOSS_CONSTRAINTS = {
    "cox": [False],
    "efron": [False],
    "Exponential": [True],
    "Weibull": [True, True],
    "Gompertz": [False, True],
}

# list out things that are available for public use
__all__ = (
    # functions and classes of this package
    "cox",
    "efron",
    "Exponential",
    "Weibull",
    "Gompertz",
)
