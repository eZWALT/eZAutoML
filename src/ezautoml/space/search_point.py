

from typing import Dict, Any, List, Optional
import yaml
from ezautoml.space.component import Component

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
        data_processors: Optional[List[Component]] = None,
        data_params_list: Optional[List[Dict[str, Any]]] = None,
        feature_processors: Optional[List[Component]] = None,
        feature_params_list: Optional[List[Dict[str, Any]]] = None,
    ):
        self.model = model
        self.model_params = model_params

        self.data_processors = data_processors or []
        self.data_params_list = data_params_list or [{} for _ in self.data_processors]

        self.feature_processors = feature_processors or []
        self.feature_params_list = feature_params_list or [{} for _ in self.feature_processors]

    def instantiate_pipeline(self):
        """
        Instantiates the pipeline in the order:
        [data_processors] -> [feature_processors] -> model
        """
        data_instances = [
            proc.instantiate(params)
            for proc, params in zip(self.data_processors, self.data_params_list)
        ]
        feature_instances = [
            proc.instantiate(params)
            for proc, params in zip(self.feature_processors, self.feature_params_list)
        ]
        model_instance = self.model.instantiate(self.model_params)

        return data_instances + feature_instances + [model_instance]

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model.name,
            "model_params": self.model_params,
            "data_processors": [
                {"name": proc.name, "params": params}
                for proc, params in zip(self.data_processors, self.data_params_list)
            ],
            "feature_processors": [
                {"name": proc.name, "params": params}
                for proc, params in zip(self.feature_processors, self.feature_params_list)
            ]
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.describe()

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def from_yaml(path: str, components: List[Component]) -> 'SearchPoint':
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        def find_component(name):
            return next(c for c in components if c.name == name)

        model = find_component(data["model"])
        model_params = data["model_params"]

        data_processors = [find_component(dp["name"]) for dp in data.get("data_processors", [])]
        data_params_list = [dp["params"] for dp in data.get("data_processors", [])]

        feature_processors = [find_component(fp["name"]) for fp in data.get("feature_processors", [])]
        feature_params_list = [fp["params"] for fp in data.get("feature_processors", [])]

        return SearchPoint(
            model=model,
            model_params=model_params,
            data_processors=data_processors,
            data_params_list=data_params_list,
            feature_processors=feature_processors,
            feature_params_list=feature_params_list
        )

    def __str__(self):
        desc = self.describe()
        return yaml.dump(desc, sort_keys=False)
    
if __name__ == "__main__":
    # Define hyperparameters
    from ezautoml.space.space import Integer, Real
    from ezautoml.space.hyperparam import Hyperparam
    from ezautoml.space.search_point import SearchPoint

    # Dummy constructors for demonstration
    class DummyModel:
        def __init__(self, **kwargs): pass

    class DummyScaler:
        def __init__(self, **kwargs): pass

    class DummyPCA:
        def __init__(self, **kwargs): pass

    model_hparams = [
        Hyperparam("n_estimators", Integer(10, 100)),
        Hyperparam("max_depth", Integer(3, 10))
    ]
    scaler_hparams = []
    pca_hparams = [Hyperparam("n_components", Real(0.5, 0.99))]

    # Create components
    model = Component("DummyModel", DummyModel, model_hparams)
    scaler = Component("StandardScaler", DummyScaler, scaler_hparams)
    pca = Component("PCA", DummyPCA, pca_hparams)

    # Sample hyperparameters
    model_params = model.sample_params()
    scaler_params = scaler.sample_params()
    pca_params = pca.sample_params()

    # Create a SearchPoint
    point = SearchPoint(
        model=model,
        model_params=model_params,
        data_processors=[scaler],
        data_params_list=[scaler_params],
        feature_processors=[pca],
        feature_params_list=[pca_params]
    )

    # Output description
    print("Sampled SearchPoint:")
    print(point)
