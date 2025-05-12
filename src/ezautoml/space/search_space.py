

# ===----------------------------------------------------------------------===#
# Search Space                                                                #
#                                                                             #
# Object that carries the whole search space composed of:                     #
# 1. Model                                                                    #
# 2.                                                                          #
# 3.                                                                          #
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

from typing import List, Optional
import random
import yaml
import importlib.resources

from ezautoml.evaluation.task import TaskType
from ezautoml.space.search_point import SearchPoint
from ezautoml.space.component import Component
# Default search spaces serialized
import ezautoml.resources.spaces as spaces


class SearchSpace:
    def __init__(
        self,
        models: List[Component],
        data_processors: Optional[List[Component]] = None,
        feature_processors: Optional[List[Component]] = None,
        task: TaskType = TaskType.BOTH,
    ):
        if not models:
            raise ValueError("SearchSpace must include at least one model.")
        
        # Validate unique component names
        all_names = [c.name for c in models + (data_processors or []) + (feature_processors or [])]
        if len(all_names) != len(set(all_names)):
            raise ValueError("Component names must be unique across all component lists.")

        self.models = models
        self.data_processors = data_processors or []
        self.feature_processors = feature_processors or []
        self.task = task

    def sample(self, seed: Optional[int] = None) -> SearchPoint:
        rng = random.Random(seed) if seed is not None else random

        model_candidates = [m for m in self.models if m.is_compatible(self.task)]
        if not model_candidates:
            raise ValueError(f"No compatible models found for task: {self.task}")
        model = rng.choice(model_candidates)

        data_proc_list = []
        data_params_list = []
        if self.data_processors:
            compatible_data = [d for d in self.data_processors if d.is_compatible(self.task)]
            if not compatible_data:
                raise ValueError(f"No compatible data processors found for task: {self.task}")
            selected_data_proc = rng.choice(compatible_data)
            data_proc_list = [selected_data_proc]
            data_params_list = [selected_data_proc.sample_params()]

        feat_proc_list = []
        feat_params_list = []
        if self.feature_processors:
            compatible_feat = [f for f in self.feature_processors if f.is_compatible(self.task)]
            if not compatible_feat:
                raise ValueError(f"No compatible feature processors found for task: {self.task}")
            selected_feat_proc = rng.choice(compatible_feat)
            feat_proc_list = [selected_feat_proc]
            feat_params_list = [selected_feat_proc.sample_params()]

        return SearchPoint(
            model=model,
            model_params=model.sample_params(),
            data_processors=data_proc_list,
            data_params_list=data_params_list,
            feature_processors=feat_proc_list,
            feature_params_list=feat_params_list,
        )

    def list_components(self) -> dict:
        return {
            "models": [m.name for m in self.models],
            "data_processors": [d.name for d in self.data_processors],
            "feature_processors": [f.name for f in self.feature_processors],
        }

    def to_yaml(self, path: str) -> None:
        full_dict = {
            "models": [m.to_dict() for m in self.models],
            "data_processors": [d.to_dict() for d in self.data_processors],
            "feature_processors": [f.to_dict() for f in self.feature_processors],
            "task": self.task.value,
        }
        with open(path, "w") as f:
            yaml.dump(full_dict, f)

    @staticmethod
    def from_yaml(path: str) -> 'SearchSpace':
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        models = [Component.from_dict(d) for d in data["models"]]
        data_procs = [Component.from_dict(d) for d in data.get("data_processors", [])]
        feat_procs = [Component.from_dict(d) for d in data.get("feature_processors", [])]
        task = TaskType[data["task"].upper()]

        return SearchSpace(models, data_procs, feat_procs, task)
    
    @staticmethod
    def from_builtin(name: str) -> 'SearchSpace':
        """Load a built-in YAML search space by name."""
        import ezautoml.resources.spaces as spaces
        # Using importlib.resources.files() to read the YAML file from package resources
        with importlib.resources.files(spaces).joinpath(f"{name}.yaml").open("r") as f:
            data = yaml.safe_load(f)

        # Parse components from the YAML data
        models = [Component.from_dict(d) for d in data["models"]]
        data_procs = [Component.from_dict(d) for d in data.get("data_processors", [])]
        feat_procs = [Component.from_dict(d) for d in data.get("feature_processors", [])]
        task = TaskType[data["task"].upper()]

        # Return the SearchSpace instance
        return SearchSpace(models, data_procs, feat_procs, task)

    def __str__(self):
        models_str = ', '.join([str(model) for model in self.models])
        data_processors_str = ', '.join([str(dp) for dp in self.data_processors]) if self.data_processors else "None"
        feature_processors_str = ', '.join([str(fp) for fp in self.feature_processors]) if self.feature_processors else "None"
        return (
            f"SearchSpace(task={self.task.name}, "
            f"models=[{models_str}], "
            f"data_processors=[{data_processors_str}], "
            f"feature_processors=[{feature_processors_str}])"
        )

if __name__ == "__main__":
    # Load a built-in search space by name
    search_space = SearchSpace.from_builtin("regression_space")

    # Print out the loaded search space
    print(f"Loaded SearchSpace: {search_space}")