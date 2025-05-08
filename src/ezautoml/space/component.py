from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, List, Dict
from enum import Enum
import inspect
from loguru import logger


from ezautoml.evaluation.task import TaskType
from ezautoml.space.space import * 
from ezautoml.space.hyperparam import Hyperparam
from ezautoml.registry import constructor_registry

# ===----------------------------------------------------------------------===#
# Abstract Component                                                          #
#                                                                             #
# This abstract class defines a component of the optimization space such as a #
# learning tool (model), data processor or other artifacts related to the auto#
# ml framework defined in the literature. It supports hierarchy of hyperparams#
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#


# To avoid doing subclasses for the moment
# TODO: use anything other than MODEL_SELECTION

# These are the common 5 steps to automate in AutoML
class Tag(Enum):
    MODEL_SELECTION = "model_selection"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_PROCESSING = "feature_processing"
    DATA_PROCESSING = "data_processing"
    OPTIMIZATION_ALGORITHM_SELECTION = "optimization_algorithm_selection"
    
class Component:
    def __init__(
        self,
        name: str,
        constructor: Callable,
        tag: Tag,
        hyperparams: List[Hyperparam] = None,
        task: TaskType = TaskType.BOTH,
        validate_interface: bool = True
    ):
        if not callable(constructor):
            raise ValueError(f"Constructor must be callable, got {constructor}")

        if not constructor_registry.has(constructor.__name__):
            logger.info(constructor_registry)
            raise ValueError(f"Constructor '{constructor.__name__}' is not registered in constructor_registry.")
        
        self.name = name
        self.constructor = constructor
        self.hyperparams = hyperparams or []
        self.task = task
        self.tag = tag

        if validate_interface:
            self._validate_interface()

    def _validate_interface(self):
        instance = self.constructor()  # Assumes default constructor works
        required_methods = []

        if self.tag == Tag.MODEL_SELECTION:
            required_methods = ["fit", "predict"]  # optionally "predict_proba"
        elif self.tag in {Tag.DATA_PROCESSING, Tag.FEATURE_PROCESSING, Tag.FEATURE_ENGINEERING}:
            required_methods = ["fit", "transform"]
        elif self.tag == Tag.OPTIMIZATION_ALGORITHM_SELECTION:
            required_methods = []

        missing = [method for method in required_methods if not hasattr(instance, method)]
        if missing:
            raise TypeError(
                f"Component '{self.name}' of tag '{self.tag.name}' is missing required methods: {missing}"
            )

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
            "tag": self.tag.value,
        }

    @classmethod
    def from_dict(cls, data: dict):
        constructor_name = data["constructor"]
        constructor = constructor_registry.get(constructor_name)
        hyperparams = [Hyperparam.from_dict(hp) for hp in data.get("hyperparams", [])]
        task = TaskType(data["task"])
        tag = Tag(data.get("tag", Tag.MODEL_SELECTION.value))
        return cls(data["name"], constructor, hyperparams, task, tag)

    def __str__(self):
        hyperparam_strs = [str(hp) for hp in self.hyperparams]
        return (
            f"Component(name='{self.name}', "
            f"task='{self.task.name}', "
            f"tag='{self.tag.name}', "
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
    rf_component = Component("RandomForest", tag=Tag.MODEL_SELECTION, constructor=RandomForestClassifier,hyperparams=rf_params)
    lr_component = Component("LogisticRegression",tag=Tag.MODEL_SELECTION, constructor=LogisticRegression, hyperparams=lr_params)

    # Feature processors
    pca_component = Component("PCA",tag=Tag.FEATURE_PROCESSING, constructor=PCA, hyperparams=[Hyperparam("n_components", Real(0.1, 0.95))])
    # Data processors
    scaler_component = Component("StandardScaler", tag=Tag.DATA_PROCESSING, constructor=StandardScaler)

    # Manually simulate a SearchSpace sampling
    all_models = [rf_component, lr_component]
    all_feature_processors = [pca_component]
    all_data_processors = [scaler_component]

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

    logger.info("Sampled Search Configuration:")
    logger.info(config)
