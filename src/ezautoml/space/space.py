from abc import ABC, abstractmethod
import random

# ===----------------------------------------------------------------------===#
# Space                                                                       #
#                                                                             #
# This abstract class defines ranges for hyperparameters of different types:  #
# Integer numbers (Natural, Integer), Real and Categorical values which can be#
# used to define the whole program search space                               #
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#


# Define Space classes
class Space(ABC):
    @abstractmethod
    def sample(self):
        """Sample a value from the space."""
        pass
    
    @abstractmethod
    def to_dict(self):
        """Serialize the space to a dictionary."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Categorical(Space):
    def __init__(self, categories):
        assert isinstance(categories, list) and len(categories) > 0, "Must provide a non-empty list of categories."
        self.categories = categories

    def contains(self, value):
        return value in self.categories

    def sample(self) -> str:
        return random.choice(self.categories)

    def to_dict(self):
        """Serialize the Categorical space to a dictionary."""
        return {
            'type': 'Categorical',
            'categories': self.categories
        }

    def __str__(self):
        return f"Categorical({self.categories})"

class Integer(Space):
    def __init__(self, low, high):
        assert isinstance(low, int) and isinstance(high, int), "Bounds must be integers."
        assert low <= high, f"Invalid Integer bounds: low={low}, high={high}"
        self.low = low
        self.high = high

    def sample(self) -> int:
        return random.randint(self.low, self.high)

    def to_dict(self):
        """Serialize the Integer space to a dictionary."""
        return {
            'type': 'Integer',
            'low': self.low,
            'high': self.high
        }

    def __str__(self):
        return f"Integer({self.low}, {self.high})"

class Real(Space):
    def __init__(self, low, high):
        assert isinstance(low, (int, float)) and isinstance(high, (int, float)), "Bounds must be numeric."
        assert low <= high, f"Invalid Real bounds: low={low}, high={high}"
        self.low = low
        self.high = high

    def sample(self) -> float:
        return random.uniform(self.low, self.high)

    def to_dict(self):
        """Serialize the Real space to a dictionary."""
        return {
            'type': 'Real',
            'low': self.low,
            'high': self.high
        }

    def __str__(self):
        return f"Real({self.low}, {self.high})"


# Example of defining a simple search space
if __name__ == "__main__":
    search_space = [
        Categorical(["xgb", "rf", "lgbm"]),
        Integer(50, 200),
        Real(0.01, 0.3),
        Categorical(["standard", "minmax", "none"]),
        Real(0.1, 1.0),
    ]

    # Function to sample from the search space
    def sample_search_space(search_space):
        sampled_params = []
        for space in search_space:
            sampled_params.append(space.sample())
        return sampled_params


    sampled_point = sample_search_space(search_space)
    print(sampled_point)
