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
class Space:
    @abstractmethod
    def sample(self):
        """Sample a value from the space."""
        pass

class Categorical(Space):
    def __init__(self, categories):
        self.categories = categories

    def contains(self, value):
        return value in self.categories
    
    def sample(self):
        return random.choice(self.categories)

class Integer(Space):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return random.randint(self.low, self.high)

class Real(Space):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return random.uniform(self.low, self.high)



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
