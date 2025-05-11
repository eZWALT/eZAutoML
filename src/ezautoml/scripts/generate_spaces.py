import yaml
from ezautoml.evaluation.task import TaskType
from ezautoml.space.component import Component, Tag
from ezautoml.space.search_space import SearchSpace
from ezautoml.space.hyperparam import Hyperparam 
from ezautoml.space.space import Integer, Real, Categorical
from ezautoml.registry import constructor_registry


serialize = True

# -----------------------------
# Define models by task
# -----------------------------
classification_model_names = [
    "RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression", "SVC",
    "KNeighborsClassifier", "DecisionTreeClassifier", "GaussianNB",
    "AdaBoostClassifier", "BaggingClassifier", "ExtraTreesClassifier",
    "XGBClassifier", "LGBMClassifier",
]

regression_model_names = [
    "RandomForestRegressor", "GradientBoostingRegressor", "Ridge", "Lasso",
    "ElasticNet", "LinearRegression", "SVR", "KNeighborsRegressor",
    "DecisionTreeRegressor", "XGBRegressor", "AdaBoostRegressor",
    "BaggingRegressor", "ExtraTreesRegressor", "LGBMRegressor",
]

# -----------------------------
# Define null components
# -----------------------------
null_components = [
    ("no_data_proc", "NoDataProcessing", TaskType.BOTH, Tag.DATA_PROCESSING),
    ("no_feat_proc", "NoFeatureProcessing", TaskType.BOTH, Tag.FEATURE_PROCESSING),
    ("no_feat_eng", "NoFeatureEngineering", TaskType.BOTH, Tag.FEATURE_ENGINEERING),
    ("no_opt_alg", "NoOptimizationAlgSelection", TaskType.BOTH, Tag.OPTIMIZATION_ALGORITHM_SELECTION)
]

# -----------------------------
# Helper functions to get registered components
# -----------------------------
def get_registered_components(model_names, task):
    components = []

    for name in model_names:
        if not constructor_registry.has(name):
            continue

        constructor = constructor_registry.get(name)

        # Shared hyperparams
        rf_tree_common = [
            Hyperparam("n_estimators", Integer(10, 1000)),
            Hyperparam("max_depth", Integer(1, 50)),
            Hyperparam("min_samples_split", Integer(2, 20)),
            Hyperparam("min_samples_leaf", Integer(1, 10))
        ]
        boosting_common = [
            Hyperparam("n_estimators", Integer(10, 1000)),
            Hyperparam("learning_rate", Real(0.01, 0.5)),
            Hyperparam("max_depth", Integer(1, 50))
        ]
        bagging_common = [
            Hyperparam("n_estimators", Integer(10, 100)),
            Hyperparam("max_samples", Real(0.1, 1.0)),
            Hyperparam("max_features", Real(0.1, 1.0))
        ]

        if name in ["RandomForestClassifier", "RandomForestRegressor"]:
            hyperparams = rf_tree_common + [Hyperparam("max_features", Categorical(["sqrt", "log2", None]))]
        elif name in ["GradientBoostingClassifier", "GradientBoostingRegressor"]:
            hyperparams = boosting_common + [Hyperparam("subsample", Real(0.5, 1.0))]
        elif name == "LogisticRegression":
            hyperparams = [
                Hyperparam("C", Real(1e-4, 100.0)),
                Hyperparam("max_iter", Integer(100, 1000)),
                Hyperparam("penalty", Categorical(["l2"]))
            ]
        elif name in ["Ridge"]:
            hyperparams = [
                Hyperparam("alpha", Real(1e-4, 100.0)),
            ]
        elif name == "Lasso":
            hyperparams = [
                Hyperparam("alpha", Real(1e-4, 10.0)),
                Hyperparam("max_iter", Integer(100, 1000))
            ]
        elif name == "ElasticNet":
            hyperparams = [
                Hyperparam("alpha", Real(1e-4, 10.0)),
                Hyperparam("l1_ratio", Real(0.0, 1.0)),
                Hyperparam("max_iter", Integer(100, 1000))
            ]
        elif name == "LinearRegression":
            hyperparams = []
        elif name == "SVC":
            hyperparams = [
                Hyperparam("C", Real(0.1, 100.0)),
                Hyperparam("kernel", Categorical(["linear", "poly", "rbf", "sigmoid"])),
                Hyperparam("gamma", Categorical(["scale", "auto"])),
                Hyperparam("degree", Integer(2, 5))
            ]
        elif name == "SVR":
            hyperparams = [
                Hyperparam("C", Real(0.1, 100.0)),
                Hyperparam("epsilon", Real(0.01, 1.0)),
                Hyperparam("kernel", Categorical(["linear", "poly", "rbf", "sigmoid"])),
                Hyperparam("gamma", Categorical(["scale", "auto"]))
            ]
        elif name in ["KNeighborsClassifier", "KNeighborsRegressor"]:
            hyperparams = [
                Hyperparam("n_neighbors", Integer(1, 50)),
                Hyperparam("weights", Categorical(["uniform", "distance"])),
                Hyperparam("leaf_size", Integer(10, 100)),
                Hyperparam("p", Integer(1, 2))
            ]
        elif name in ["DecisionTreeClassifier", "DecisionTreeRegressor"]:
            hyperparams = [
                Hyperparam("criterion", Categorical(["gini", "entropy", "log_loss"] if "Classifier" in name else ["squared_error", "friedman_mse", "absolute_error", "poisson"])),
                Hyperparam("max_depth", Integer(1, 50)),
                Hyperparam("min_samples_split", Integer(2, 20)),
                Hyperparam("min_samples_leaf", Integer(1, 10))
            ]
        elif name == "GaussianNB":
            hyperparams = []
        elif name == "AdaBoostClassifier" or name == "AdaBoostRegressor":
            hyperparams = [
                Hyperparam("n_estimators", Integer(10, 1000)),
                Hyperparam("learning_rate", Real(0.01, 2.0)),
            ]
        elif name == "BaggingClassifier" or name == "BaggingRegressor":
            hyperparams = bagging_common
        elif name == "ExtraTreesClassifier" or name == "ExtraTreesRegressor":
            hyperparams = rf_tree_common
        elif name in ["XGBClassifier", "XGBRegressor"]:
            hyperparams = [
                Hyperparam("n_estimators", Integer(10, 1000)),
                Hyperparam("learning_rate", Real(0.01, 0.3)),
                Hyperparam("max_depth", Integer(3, 15)),
                Hyperparam("min_child_weight", Integer(1, 10)),
                Hyperparam("subsample", Real(0.5, 1.0)),
                Hyperparam("colsample_bytree", Real(0.5, 1.0)),
                Hyperparam("gamma", Real(0.0, 5.0)),  # minimum loss reduction
                Hyperparam("reg_alpha", Real(0.0, 10.0)),  # L1 regularization
                Hyperparam("reg_lambda", Real(0.0, 10.0)),  # L2 regularization
            ]
        elif name in ["LGBMClassifier", "LGBMRegressor"]:
            hyperparams = [
                Hyperparam("n_estimators", Integer(10, 1000)),
                Hyperparam("learning_rate", Real(0.001, 0.3)),
                Hyperparam("num_leaves", Integer(20, 150)),
                Hyperparam("max_depth", Integer(-1, 30)),  # -1 means no limit
                Hyperparam("min_child_samples", Integer(5, 100)),
                Hyperparam("subsample", Real(0.5, 1.0)),  # bagging fraction
                Hyperparam("colsample_bytree", Real(0.5, 1.0)),  # feature fraction
                Hyperparam("reg_alpha", Real(0.0, 10.0)),  # L1 regularization
                Hyperparam("reg_lambda", Real(0.0, 10.0)),  # L2 regularization
            ]            
        else:
            hyperparams = []

        components.append(Component(
            name=name,
            constructor=constructor,
            task=task,
            tag=Tag.MODEL_SELECTION,
            hyperparams=hyperparams
        ))

    return components


def get_null_components():
    components = []
    for name, registry_name, task, tag in null_components:
        if constructor_registry.has(registry_name):
            constructor = constructor_registry.get(registry_name)
            components.append(Component(name=name, constructor=constructor, task=task, tag=tag))
    return components

# -----------------------------
# Build model, data, and feature components for each task
# -----------------------------
classification_models = get_registered_components(classification_model_names, TaskType.CLASSIFICATION)
regression_models = get_registered_components(regression_model_names, TaskType.REGRESSION)

# **Only include ONE null component for each task**
data_processors = [get_null_components()[0]]  # Only the NoDataProcessing component
feature_processors = [get_null_components()[1]]  # Only the NoFeatureProcessing component

# -----------------------------
# Build search spaces with models and hyperparameters
# -----------------------------
classification_space = SearchSpace(
    models=classification_models,                # Only the classification models with hyperparameters
    data_processors=data_processors,              # Just one data processor
    feature_processors=feature_processors,        # Just one feature processor
    task=TaskType.CLASSIFICATION
)

regression_space = SearchSpace(
    models=regression_models,                     # Only the regression models with hyperparameters
    data_processors=data_processors,              # Just one data processor
    feature_processors=feature_processors,        # Just one feature processor
    task=TaskType.REGRESSION
)


# -----------------------------
# Serialize to YAML (optional)
# -----------------------------

if serialize:
    regression_space.to_yaml(path="./regression_space.yaml")
    classification_space.to_yaml(path="./classification_space.yaml")
