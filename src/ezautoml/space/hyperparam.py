# Hyperparameter class (to sample and store hyperparameters)
from ezautoml.space.space import Integer, Real, Categorical, Space

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

    def sample(self) -> any:
        """Sample a value from the hyperparameter space."""
        return self.space.sample()

    def to_dict(self) -> dict:
        """Serialize a hyperparameter to a dictionary."""
        return {
            'name': self.name,
            'type': self.space.__class__.__name__,
            'space': self.space.__dict__
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Hyperparam':
        space_type = data['type']
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