

from typing import Dict, Any, List
import yaml
from ezautoml.space import Component

# ===----------------------------------------------------------------------===#
# Search Point (Slice of Seach Space)                                         #
#                                                                             #
# Lol                                                                         #
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

class SearchPoint:
    def __init__(
        self,
        model: Component,
        model_params: Dict[str, Any],
        data_proc: Component,
        data_params: Dict[str, Any],
        feat_proc: Component,
        feat_params: Dict[str, Any],
    ):
        self.model = model
        self.model_params = model_params
        self.data_proc = data_proc
        self.data_params = data_params
        self.feat_proc = feat_proc
        self.feat_params = feat_params

    def instantiate_pipeline(self):
        # Instantiate the actual sklearn-like pipeline
        model_instance = self.model.instantiate(self.model_params)
        data_instance = self.data_proc.instantiate(self.data_params)
        feat_instance = self.feat_proc.instantiate(self.feat_params)
        return (data_instance, feat_instance, model_instance)

    def describe(self):
        return {
            "model": self.model.name,
            "model_params": self.model_params,
            "data_processor": self.data_proc.name,
            "data_params": self.data_params,
            "feature_processor": self.feat_proc.name,
            "feat_params": self.feat_params
        }
        
    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def from_yaml(path: str, components: List[Component]) -> 'SearchPoint':
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        def find_component(name, comps):
            return next(c for c in comps if c.name == name)

        model = find_component(data["model"], components)
        data_proc = find_component(data["data_processor"], components)
        feat_proc = find_component(data["feature_processor"], components)

        return SearchPoint(
            model=model,
            model_params=data["model_params"],
            data_proc=data_proc,
            data_params=data["data_params"],
            feat_proc=feat_proc,
            feat_params=data["feature_params"]
        )


