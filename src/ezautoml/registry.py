# Global Constructor Registry to serialize/deserialize safely 
from dataclasses import dataclass, field
from typing import Callable, Dict

# ===----------------------------------------------------------------------===#
# Constructor Registry (Configuration file of allowed modules)                #
#                                                                             #
# Can be used to safely serialize and deserialize constructors but also to    #
# limit in a configuration file fashion which models/processors are supported #
#                                                                             #
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

@dataclass
class ConstructorRegistry:
    registry: Dict[str, Callable] = field(default_factory=dict)

    def register(self, constructor: Callable):
        """Registers a constructor by its __name__."""
        name = constructor.__name__
        if name in self.registry:
            raise ValueError(f"Constructor '{name}' is already registered.")
        self.registry[name] = constructor
        return constructor  # allows use as a decorator

    def get(self, name: str) -> Callable:
        """Retrieves a constructor by name."""
        if name not in self.registry:
            raise ValueError(f"Constructor '{name}' not found in registry.")
        return self.registry[name]

    def has(self, name: str) -> bool:
        return name in self.registry

    def list(self):
        """Returns a list of all registered constructor names."""
        return list(self.registry)
    
    
###############################################################################
###############################################################################
################### Instantiate registry structure ############################
###############################################################################
###############################################################################

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    LinearRegression
)
from sklearn.svm import (
    SVC, SVR
)
from sklearn.neighbors import (
    KNeighborsClassifier, KNeighborsRegressor
)
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor
)
from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Importing other popular models with scikit-learn like API
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


# NoFeatureEngineering: Does nothing when called, just returns the input
class NoFeatureEngineering:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X


# NoFeatureProcessing: Does nothing when called, just returns the input
class NoFeatureProcessing:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

# NoDataProcessing: Does nothing, just returns the input
class NoDataProcessing:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X

# NoOptimizationAlgSelection: A placeholder class to simulate no optimization algorithm selection
class NoOptimizationAlgSelection:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X


constructor_registry = ConstructorRegistry()

# Register all constructors once here
for constructor in [
    # -----------------------------------------------------
    # 1. Classification Models (Expanded)
    # -----------------------------------------------------
    RandomForestClassifier,
    GradientBoostingClassifier,
    LogisticRegression,
    SVC, 
    KNeighborsClassifier,
    DecisionTreeClassifier,
    GaussianNB,
    MultinomialNB,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,  # Bagging-based model
    XGBClassifier,
    LGBMClassifier,
    CatBoostClassifier,

    # -----------------------------------------------------
    # 2. Regression Models (Expanded)
    # -----------------------------------------------------
    RandomForestRegressor,
    GradientBoostingRegressor,
    Ridge,
    Lasso,
    ElasticNet,
    LinearRegression,
    SVR,
    KNeighborsRegressor,
    DecisionTreeRegressor,
    XGBRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,  # Bagging-based model
    LGBMRegressor,
    CatBoostRegressor,

    # -----------------------------------------------------
    # 3. Feature processing components (Top 5)
    # -----------------------------------------------------
    KMeans,
    PCA,

    # -----------------------------------------------------
    # 4. Data processing components (Top 5)
    # -----------------------------------------------------
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    
    # -----------------------------------------------------
    # 5. Null components
    # -----------------------------------------------------
    NoFeatureEngineering,
    NoDataProcessing, 
    NoFeatureProcessing,
    NoOptimizationAlgSelection
    
]:
    constructor_registry.register(constructor)
