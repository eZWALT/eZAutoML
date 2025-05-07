from abc import ABC
from typing import Callable, List
from ezautoml.evaluation.task import TaskType
from ezautoml.space.space import * 
from ezautoml.space.hyperparam import Hyperparam


# TODO: perform some validation of allowed models, preprocessors...


# ===----------------------------------------------------------------------===#
# Abstract Component                                                          #
#                                                                             #
# This abstract class defines a component of the optimization space such as a #
# learning tool (model), data processor or other artifacts related to the auto#
# ml framework defined in the literature. It supports hierarchy of hyperparams#
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

# dummy NoOp constructor (Just for testing this shouldn't be used regularly)
# alternatively empty lambda can be used
def NoOp():
    return None

class Component:
    def __init__(
        self,
        name: str,
        constructor: Callable,
        hyperparams: List[Hyperparam] = None,
        task: TaskType = TaskType.BOTH,
    ):
        """
        Represents a model or processor with its constructor, hyperparameters, and task compatibility.
        """
        if not callable(constructor):
            raise ValueError(f"Constructor must be callable, got {constructor}")
        
        self.name = name
        self.constructor = constructor
        self.hyperparams = hyperparams or []
        self.task = task

    def sample_params(self) -> dict:
        return {hp.name: hp.sample() for hp in self.hyperparams}

    def instantiate(self, params: dict):
        return self.constructor(**params)

    def is_compatible(self, task: TaskType) -> bool:
        return self.task == TaskType.BOTH or self.task == task

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "constructor": self.constructor.__name__,
            "hyperparams": [hp.to_dict() for hp in self.hyperparams],
            "task": self.task.value,
        }

    @classmethod
    def from_dict(cls, data: dict):
        constructor_name = data["constructor"]
        constructor = globals().get(constructor_name)
        if constructor is None:
            raise ValueError(f"Constructor '{constructor_name}' not found in global scope")
        
        hyperparams = [Hyperparam.from_dict(hp) for hp in data.get("hyperparams", [])]
        task = TaskType(data["task"])
        return cls(data["name"], constructor, hyperparams, task)

    def __str__(self):
        hyperparam_strs = [str(hp) for hp in self.hyperparams]
        return (
            f"Component(name='{self.name}', "
            f"task='{self.task.name}', "
            f"constructor='{self.constructor.__name__}', "
            f"hyperparams=[{', '.join(hyperparam_strs)}])"
        )


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Define hyperparameters for RandomForest
    rf_params = [
        Hyperparam("n_estimators", Integer(50, 150)),
        Hyperparam("max_depth", Integer(5, 20))
    ]

    # Define hyperparameters for LogisticRegression
    lr_params = [
        Hyperparam("C", Real(0.01, 10)),
        Hyperparam("penalty", Categorical(["l2", "none"]))
    ]

    # Model components
    rf_component = Component("RandomForest", RandomForestClassifier, rf_params)
    lr_component = Component("LogisticRegression", LogisticRegression, lr_params)

    # Feature processors
    pca_component = Component("PCA", PCA, [Hyperparam("n_components", Real(0.1, 0.95))])
    no_pca_component = Component("NoPCA", lambda: None, [])  # Dummy for "no feature processing"

    # Data processors
    scaler_component = Component("StandardScaler", StandardScaler)
    no_scaler_component = Component("NoScaler", lambda: None)

    # Manually simulate a SearchSpace sampling
    all_models = [rf_component, lr_component]
    all_feature_processors = [pca_component, no_pca_component]
    all_data_processors = [scaler_component, no_scaler_component]

    # Simulate hierarchical sampling
    chosen_model = random.choice(all_models)
    chosen_feature_proc = random.choice(all_feature_processors)
    chosen_data_proc = random.choice(all_data_processors)

    config = {
        "model": chosen_model.name,
        "model_params": chosen_model.sample_params(),
        "feature_processor": chosen_feature_proc.name,
        "feature_params": chosen_feature_proc.sample_params(),
        "data_processor": chosen_data_proc.name,
        "data_params": chosen_data_proc.sample_params(),
    }

    print("Sampled Search Configuration:")
    print(config)
    print("\n\n The Component str is:")
    print(scaler_component)
