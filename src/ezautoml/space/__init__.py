from abc import ABC
from ezautoml.evaluation.task import TaskType
from ezautoml.space.space import * 
from ezautoml.space.hyperparam import Hyperparam

# ===----------------------------------------------------------------------===#
# Abstract Component                                                          #
#                                                                             #
# This abstract class defines a component of the optimization space such as a #
# learning tool (model), data processor or other artifacts related to the auto#
# ml framework defined in the literature. It supports hierarchy of hyperparams#
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

class Component:
    def __init__(self, name: str, constructor, hyperparams=None, task: TaskType = TaskType.BOTH):
        """
        Initializes a Component.
        
        :param name: Name of the component (e.g., 'RandomForest', 'PCA', etc.).
        :param constructor: A function or callable used to instantiate the model or processor (e.g., RandomForestClassifier, PCA).
        :param hyperparams: A list of Hyperparam objects associated with this component.
        :param task: The task type for which this component is compatible, defaults to `TaskType.BOTH`.
        """
        self.name = name
        self.constructor = constructor
        self.hyperparams = hyperparams or []
        self.task = task  # Can be TaskType.CLASSIFICATION, TaskType.REGRESSION, or TaskType.BOTH for compatibility with both

    def sample_params(self) -> dict:
        """
        Samples hyperparameters for this component based on its space.

        :return: A dictionary of sampled hyperparameters for the component.
        """
        return {hp.name: hp.sample() for hp in self.hyperparams}

    def instantiate(self, params: dict):
        """
        Instantiates the component (e.g., creates a model or processor) with the provided hyperparameters.

        :param params: A dictionary of hyperparameters to set when instantiating the component.
        :return: The instantiated model or processor.
        """
        return self.constructor(**params)

    def is_compatible(self, task: TaskType) -> bool:
        """
        Checks whether this component is compatible with the given task type.

        :param task: The task type to check compatibility against (e.g., CLASSIFICATION, REGRESSION).
        :return: True if the component is compatible with the task type, False otherwise.
        """
        # If the component is marked as compatible with both task types, it's always compatible
        return self.task == TaskType.BOTH or self.task == task


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
