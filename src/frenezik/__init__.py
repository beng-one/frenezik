"""|----- frenezik -----|"""

__version__ = "0.1.0"

# Import of main class
from ._residuals import Residuals
from ._target_predictors import TargetPredictors
from ._regularization import Regularization

# Export
__all__ = ['Residuals', 'TargetPredictors', 'Regularization']
