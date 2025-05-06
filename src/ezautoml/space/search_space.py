

# ===----------------------------------------------------------------------===#
# Search Space                                                                #
#                                                                             #
# Object that carries the whole search space composed of:                     #
# 1. Model                                                                    #
# 2.                                                                          #
# 3.                                                                          #
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
        task: TaskType,
    ):
        self.models = models
        self.data_processors = data_processors
        self.feature_processors = feature_processors
        self.task = task

    def sample(self) -> SearchPoint:
        model = random.choice([m for m in self.models if m.is_compatible(self.task)])
        data_proc = random.choice([d for d in self.data_processors if d.is_compatible(self.task)])
        feat_proc = random.choice([f for f in self.feature_processors if f.is_compatible(self.task)])

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
            "task": self.task.value
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
        task = TaskType[data["task"].upper()]  # Convert string back to TaskType

        return SearchSpace(models, data_procs, feat_procs, task)
    


if __name__ == "__main__":
    from ezautoml.space import Component
    from ezautoml.space.search_space import SearchSpace
    from ezautoml.evaluation.task import TaskType
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA, FastICA
    # Define Components with appropriate constructors and task types
    models = [
        Component("rf", RandomForestClassifier, task=TaskType.BOTH),  # Example: RandomForest is compatible with both
        Component("logreg", LogisticRegression, task=TaskType.CLASSIFICATION),
    ]
    
    data_processors = [
        Component("imputation", SimpleImputer, task=TaskType.BOTH),  # SimpleImputer can be used for both tasks
        Component("scaling", StandardScaler, task=TaskType.CLASSIFICATION),
    ]
    
    feature_processors = [
        Component("pca", PCA, task=TaskType.BOTH),
        Component("ica", FastICA, task=TaskType.CLASSIFICATION),
    ]

    # Create a SearchSpace instance
    task_type = TaskType.CLASSIFICATION  # Change to REGRESSION for regression testing
    search_space = SearchSpace(models, data_processors, feature_processors, task_type)

    # Sample a SearchPoint
    search_point = search_space.sample()
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
    new_search_point = loaded_search_space.sample()
    print(f"Sampled SearchPoint from loaded SearchSpace for task '{task_type.value}':")
    print(new_search_point)
