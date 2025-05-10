from ezautoml.evaluation.task import TaskType
from ezautoml.space.component import Component
from ezautoml.space.search_space import SearchSpace
from ezautoml.registry import constructor_registry

# -----------------------------
# Define models by task
# -----------------------------
classification_models = [
    "RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression", "SVC",
    "KNeighborsClassifier", "DecisionTreeClassifier", "GaussianNB", "MultinomialNB",
    "AdaBoostClassifier", "BaggingClassifier", "ExtraTreesClassifier",
    "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"
]

regression_models = [
    "RandomForestRegressor", "GradientBoostingRegressor", "Ridge", "Lasso",
    "ElasticNet", "LinearRegression", "SVR", "KNeighborsRegressor",
    "DecisionTreeRegressor", "XGBRegressor", "AdaBoostRegressor",
    "BaggingRegressor", "ExtraTreesRegressor", "LGBMRegressor", "CatBoostRegressor"
]

# -----------------------------
# Define null components
# -----------------------------
null_components = [
    ("no_data_proc", "NoDataProcessing", TaskType.BOTH),
    ("no_feat_proc", "NoFeatureProcessing", TaskType.BOTH),
    ("no_feat_eng", "NoFeatureEngineering", TaskType.BOTH),
    ("no_opt_alg", "NoOptimizationAlgSelection", TaskType.BOTH)
]

def get_registered_components(model_names, task):
    components = []
    for name in model_names:
        if constructor_registry.has(name):
            constructor = constructor_registry.get(name)
            components.append(Component(name=name, constructor=constructor, task=task))
    return components

def get_null_components():
    components = []
    for name, registry_name, task in null_components:
        if constructor_registry.has(registry_name):
            constructor = constructor_registry.get(registry_name)
            components.append(Component(name=name, constructor=constructor, task=task))
    return components

# Build model, data, and feature components for each task
classification_models = get_registered_components(classification_models, TaskType.CLASSIFICATION)
regression_models = get_registered_components(regression_models, TaskType.REGRESSION)

data_processors = get_null_components()
feature_processors = get_null_components()

# -----------------------------
# Build search spaces
# -----------------------------
classification_space = SearchSpace(
    models=classification_models,
    data_processors=data_processors,
    feature_processors=feature_processors,
    task=TaskType.CLASSIFICATION
)

regression_space = SearchSpace(
    models=regression_models,
    data_processors=data_processors,
    feature_processors=feature_processors,
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
