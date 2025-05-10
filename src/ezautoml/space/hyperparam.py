# Hyperparameter class (to sample and store hyperparameters)
from ezautoml.space.space import Integer, Real, Categorical, Space
from typing import Union

# ===----------------------------------------------------------------------===#
# Space                                                                       #
#                                                                             #
# This abstract class defines ranges for hyperparameters of different types:  #
# Integer numbers (Natural, Integer), Real and Categorical values which can be# 
# used to define the whole program search space                               #
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

class Hyperparam:
    def __init__(self, name: str, space: Space):
        self.name = name
        self.space = space  # Space defines the range (could be Categorical, Integer, or Real)

    def sample(self) -> Union[str, int, float]:
        """Sample a value from the hyperparameter space."""
        return self.space.sample()

    def to_dict(self) -> dict:
        """Serialize a hyperparameter to a dictionary, using the space's to_dict."""
        return {
            'name': self.name,
            'space': self.space.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Hyperparam':
        space_type = data['space']['type']
        space_data = data['space']
        if space_type == 'Integer':
            return cls(data['name'], Integer(space_data['low'], space_data['high']))
        elif space_type == 'Real':
            return cls(data['name'], Real(space_data['low'], space_data['high']))
        elif space_type == 'Categorical':
            return cls(data['name'], Categorical(space_data['categories']))
        return None
    
    def __str__(self):
        return f"Hyperparam(name={self.name}, space={self.space})"
    
    
    # Example of defining a simple search space
if __name__ == "__main__":
    # Define some hyperparameters
    hyperparameters = [
        Hyperparam("model", Categorical(["xgb", "rf", "lgbm"])),
        Hyperparam("n_estimators", Integer(50, 200)),
        Hyperparam("learning_rate", Real(0.01, 0.3)),
        Hyperparam("scaler", Categorical(["standard", "minmax", "none"])),
        Hyperparam("max_depth", Integer(1, 10)),
    ]

    # Function to sample from the search space
    def sample_search_space(hyperparameters):
        sampled_params = {}
        for hp in hyperparameters:
            sampled_params[hp.name] = hp.sample()
        return sampled_params

    sampled_point = sample_search_space(hyperparameters)
    print("Sampled Point:", sampled_point)

    # Serialize the hyperparameters to a dictionary
    hyperparam_dicts = [hp.to_dict() for hp in hyperparameters]
    print("Serialized Hyperparameters:", hyperparam_dicts)