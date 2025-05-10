

from typing import Dict, Any, List, Optional
import yaml
from ezautoml.space.component import Component
from ezautoml.results.trial import Trial  
from ezautoml.space.hyperparam import Hyperparam

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
        model_params: Dict[str,Hyperparam],
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

        # stores the evaluation result
        self.result: Optional[Trial] = None  

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
            # LOL
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
        d = self.describe()
        if self.result:
            d["result"] = self.result.to_dict()
        return d

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

        sp = SearchPoint(
            model=model,
            model_params=model_params,
            data_processors=data_processors,
            data_params_list=data_params_list,
            feature_processors=feature_processors,
            feature_params_list=feature_params_list
        )

        if "result" in data:
            sp.result = Trial.from_dict(data["result"])

        return sp


    def __str__(self):
        desc = self.describe()
        return yaml.dump(desc, sort_keys=False)

if __name__ == "__main__":
    import yaml
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    from ezautoml.space.search_point import SearchPoint
    from ezautoml.space.component import Component
    from ezautoml.space.space import Integer, Real
    from ezautoml.space.hyperparam import Hyperparam
    from ezautoml.results.trial import Trial  # adjust path if needed

    # Define Components using real sklearn classes
    model = Component(
        "RandomForestClassifier",
        RandomForestClassifier,
        [
            Hyperparam("n_estimators", Integer(10, 100)),
            Hyperparam("max_depth", Integer(3, 10)),
        ]
    )

    scaler = Component("StandardScaler", StandardScaler, [])

    pca = Component(
        "PCA",
        PCA,
        [
            Hyperparam("n_components", Real(0.5, 0.99)),
        ]
    )

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
        feature_params_list=[pca_params],
    )

    # Assign a dummy trial result
    point.result = Trial(
        seed=123,
        model_name=model.name,
        optimizer_name="RandomSearch",
        evaluation={"accuracy": 0.87, "f1_score": 0.84},
        duration=12.3
    )

    # Serialize to YAML
    yaml_path = "search_point.yaml"
    point.to_yaml(yaml_path)

    # Deserialize from YAML
    loaded = SearchPoint.from_yaml(yaml_path, [model, scaler, pca])

    # Print loaded SearchPoint and Trial
    print("Restored SearchPoint:")
    print(loaded)

    if loaded.result:
        print("\nRestored Trial:")
        print(loaded.result)