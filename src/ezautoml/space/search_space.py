

# ===----------------------------------------------------------------------===#
# Search Space                                                                #
#                                                                             #
# Object that carries the whole search space composed of:                     #
# 1. Model
# 2. 
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

from typing import List
import random
import yaml

from ezautoml.evaluation.task import TaskType
from ezautoml.space import Component
from ezautoml.space.search_point import SearchPoint

class SearchSpace:
    def __init__(
        self,
        models: List[Component],
        data_processors: List[Component],
        feature_processors: List[Component],
    ):
        self.models = models
        self.data_processors = data_processors
        self.feature_processors = feature_processors

    def sample(self, task: TaskType) -> SearchPoint:
        model = random.choice([m for m in self.models if m.is_compatible(task)])
        data_proc = random.choice([d for d in self.data_processors if d.is_compatible(task)])
        feat_proc = random.choice([f for f in self.feature_processors if f.is_compatible(task)])

        return SearchPoint(
            model=model,
            model_params=model.sample_params(),
            data_proc=data_proc,
            data_params=data_proc.sample_params(),
            feat_proc=feat_proc,
            feat_params=feat_proc.sample_params(),
        )

    def to_yaml(self, path: str) -> None:
        full_dict = {
            "models": [m.to_dict() for m in self.models],
            "data_processors": [d.to_dict() for d in self.data_processors],
            "feature_processors": [f.to_dict() for f in self.feature_processors],
        }
        with open(path, "w") as f:
            yaml.dump(full_dict, f)

    @staticmethod
    def from_yaml(path: str) -> 'SearchSpace':
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        models = [Component.from_dict(d) for d in data["models"]]
        data_procs = [Component.from_dict(d) for d in data["data_processors"]]
        feat_procs = [Component.from_dict(d) for d in data["feature_processors"]]

        return SearchSpace(models, data_procs, feat_procs)
    

# Main testing script
if __name__ == "__main__":
    models = [
        Component("rf", [TaskType.CLASSIFICATION, TaskType.REGRESSION]),
        Component("xgb", [TaskType.CLASSIFICATION]),
        Component("logreg", [TaskType.CLASSIFICATION]),
    ]
    data_processors = [
        Component("imputation", [TaskType.CLASSIFICATION, TaskType.REGRESSION]),
        Component("scaling", [TaskType.CLASSIFICATION]),
    ]
    feature_processors = [
        Component("pca", [TaskType.CLASSIFICATION, TaskType.REGRESSION]),
        Component("ica", [TaskType.CLASSIFICATION]),
    ]

    # Create a SearchSpace instance
    search_space = SearchSpace(models, data_processors, feature_processors)

    # Sample a SearchPoint
    task_type = TaskType.CLASSIFICATION  # Change to REGRESSION for regression testing
    search_point = search_space.sample(task_type)
    print(f"Sampled SearchPoint for task '{task_type.value}':")
    print(search_point)

    # Test serialization to YAML
    yaml_file = "search_space.yaml"
    search_space.to_yaml(yaml_file)
    print(f"SearchSpace serialized to {yaml_file}")

    # Test deserialization from YAML
    loaded_search_space = SearchSpace.from_yaml(yaml_file)
    print(f"Deserialized SearchSpace from {yaml_file}:")
    print(loaded_search_space)

    # Sample a new SearchPoint after loading from YAML
    new_search_point = loaded_search_space.sample(task_type)
    print(f"Sampled SearchPoint from loaded SearchSpace for task '{task_type.value}':")
    print(new_search_point)