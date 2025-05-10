import yaml
from ezautoml.evaluation.task import TaskType
from ezautoml.space.component import Component, Tag
from ezautoml.space.search_space import SearchSpace
from ezautoml.space.hyperparam import Hyperparam 
from ezautoml.space.space import Integer, Real, Categorical
from ezautoml.registry import constructor_registry

# -----------------------------
# Define models by task
# -----------------------------
classification_model_names = [
    "RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression", "SVC",
    "KNeighborsClassifier", "DecisionTreeClassifier", "GaussianNB", "MultinomialNB",
    "AdaBoostClassifier", "BaggingClassifier", "ExtraTreesClassifier",
    "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"
]

regression_model_names = [
    "RandomForestRegressor", "GradientBoostingRegressor", "Ridge", "Lasso",
    "ElasticNet", "LinearRegression", "SVR", "KNeighborsRegressor",
    "DecisionTreeRegressor", "XGBRegressor", "AdaBoostRegressor",
    "BaggingRegressor", "ExtraTreesRegressor", "LGBMRegressor", "CatBoostRegressor"
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
        if constructor_registry.has(name):
            constructor = constructor_registry.get(name)
            
            # Define hyperparameters for each model
            if name == "RandomForestClassifier" or name == "RandomForestRegressor":
                hyperparams = [
                    Hyperparam("n_estimators", Integer(10, 100)),
                    Hyperparam("max_depth", Integer(3, 15)),
                    Hyperparam("min_samples_split", Integer(2, 10)),
                    Hyperparam("min_samples_leaf", Integer(1, 5)),
                ]
            elif name == "GradientBoostingClassifier" or name == "GradientBoostingRegressor":
                hyperparams = [
                    Hyperparam("n_estimators", Integer(10, 100)),
                    Hyperparam("learning_rate", Real(0.01, 0.3)),
                    Hyperparam("max_depth", Integer(3, 15)),
                ]
            elif name == "LogisticRegression":
                hyperparams = [
                    Hyperparam("C", Real(0.01, 10)),
                    Hyperparam("max_iter", Integer(50, 500)),
                    Hyperparam("solver", Categorical(["liblinear", "saga", "lbfgs"])),
                ]
            elif name == "SVC":
                hyperparams = [
                    Hyperparam("C", Real(0.01, 10)),
                    Hyperparam("kernel", Categorical(["linear", "poly", "rbf", "sigmoid"])),
                    Hyperparam("gamma", Real(0.01, 1)),
                ]
            elif name == "KNeighborsClassifier" or name == "KNeighborsRegressor":
                hyperparams = [
                    Hyperparam("n_neighbors", Integer(3, 15)),
                    Hyperparam("leaf_size", Integer(10, 50)),
                ]
            elif name == "DecisionTreeClassifier" or name == "DecisionTreeRegressor":
                hyperparams = [
                    Hyperparam("max_depth", Integer(1, 100)),
                    Hyperparam("min_samples_split", Integer(2, 10)),
                    Hyperparam("min_samples_leaf", Integer(1, 5)),
                ]
            elif name == "GaussianNB":
                hyperparams = []  # No hyperparameters typically
            elif name == "MultinomialNB":
                hyperparams = [
                    Hyperparam("alpha", Real(0.01, 10)),
                ]
            elif name == "AdaBoostClassifier" or name == "AdaBoostRegressor":
                hyperparams = [
                    Hyperparam("n_estimators", Integer(10, 100)),
                    Hyperparam("learning_rate", Real(0.01, 1)),
                ]
            elif name == "BaggingClassifier" or name == "BaggingRegressor":
                hyperparams = [
                    Hyperparam("n_estimators", Integer(10, 100)),
                    Hyperparam("max_samples", Real(0.1, 1)),
                    Hyperparam("max_features", Real(0.1, 1)),
                ]
            elif name == "ExtraTreesClassifier" or name == "ExtraTreesRegressor":
                hyperparams = [
                    Hyperparam("n_estimators", Integer(10, 100)),
                    Hyperparam("max_depth", Integer(3, 15)),
                    Hyperparam("min_samples_split", Integer(2, 10)),
                    Hyperparam("min_samples_leaf", Integer(1, 5)),
                ]
            elif name == "XGBClassifier" or name == "XGBRegressor":
                hyperparams = [
                    Hyperparam("n_estimators", Integer(10, 100)),
                    Hyperparam("learning_rate", Real(0.01, 0.3)),
                    Hyperparam("max_depth", Integer(3, 15)),
                ]
            elif name == "LGBMClassifier" or name == "LGBMRegressor":
                hyperparams = [
                    Hyperparam("n_estimators", Integer(10, 100)),
                    Hyperparam("learning_rate", Real(0.01, 0.3)),
                    Hyperparam("max_depth", Integer(3, 15)),
                ]
            elif name == "CatBoostClassifier" or name == "CatBoostRegressor":
                hyperparams = [
                    Hyperparam("iterations", Integer(10, 100)),
                    Hyperparam("learning_rate", Real(0.01, 0.3)),
                    Hyperparam("depth", Integer(3, 15)),
                ]
            else:
                hyperparams = []  # Default to no hyperparameters

            # Add component with hyperparameters
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
# Output and sampling
# -----------------------------
print("📘 Classification SearchSpace:")
print(classification_space)
print("\n🎯 Sample SearchPoint (Classification):")
print(classification_space.sample().describe())

print("\n📗 Regression SearchSpace:")
print(regression_space)
print("\n🎯 Sample SearchPoint (Regression):")
print(regression_space.sample().describe())

# -----------------------------
# Serialize to YAML (optional)
# -----------------------------
regression_space.to_yaml(path="./regression_space.yaml")
classification_space.to_yaml(path="./classification_space.yaml")
