![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)
![Stars](https://img.shields.io/github/stars/eZWALT/eZAutoML?style=flat)
![Forks](https://img.shields.io/github/forks/eZWALT/eZAutoML?style=flat)
![Last Commit](https://img.shields.io/github/last-commit/eZWALT/eZAutoML?style=flat)
![Commit Activity](https://img.shields.io/github/commit-activity/m/eZWALT/eZAutoML?style=flat)
![Docs](https://img.shields.io/badge/docs-latest-blue)

<!---
![Version](https://img.shields.io/github/v/tag/eZWALT/eZAutoML?style=flat)
![PyPI Downloads](https://img.shields.io/pypi/dm/eZAutoML?style=flat)
-->

# eZAutoML 

<!---
![](./resources/logo_red_transparent.png)
-->
<p align="center">
  <img src="./resources/logo_transparent.png" alt="eZAutoML Logo" width="300"/>
</p>

## Overview

`eZAutoML` is a framework designed to make Automated Machine Learning (AutoML) accessible to everyone. It provides an incredible easy to use interface based on Scikit-Learn API to build modelling pipelines with minimal effort.

The framework is built around a few core concepts:

1. **Optimizers**: Black-box optimization methods for hyperparameters.
2. **Easy Tabular Pipelines**: Simple domain-specific language to describe pipelines for preprocessing and model training.
3. **Scheduling**: Work in progress; this feature enables horizontal scalability from a single computer to datacenters by using airflow executors.

## Installation 

### Package Distribution 

The latest version of `eZAutoML` can be installed via **PyPI** or from source.

```bash 
pip install ezautoml
ezautoml --help
```

### Install from source
To install from source, you can clone this repo and install with `pip`:

```
pip install -e .
```

## Usage

### Command Line Interface 

Not only it can be used programatically but we provide an extremely lightweight CLI api to instantiate tabular AutoML pipelines with just a single command, for example: 

```bash
ezautoml --dataset data/smoking.csv --target smoking --task classification --trials 10 --verbose   
```

Options:
- dataset: Path to the dataset file (CSV, parquet...)
- target: The target column name for prediction
- task: Task type: classification/c or regression/r
- search: Black-box optimization algorithm to perform
- output: Directory to save the output models/results
- trials: Maximum number of trials inside an optimiation algorithm
- verbose: Increase logging verbosity 
- version: Show the current version 

For more detailed help, use:

```bash
ezautoml --help
```

There are future features that are still a work-in-progress and will be enabled in the future such as scheduling, metalearning, pipelines...

### Python Script

You can also use eZAutoML within Python scripts (though this feature is still being developed). This will allow you to work through Python code or via custom pipelines in the future.

```python
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score
    from ezautoml.model import eZAutoML
    from ezautoml.space.search_space import SearchSpace
    from ezautoml.evaluation.metric import MetricSet, Metric
    from ezautoml.evaluation.task import TaskType
    from ezautoml.optimization.optimizers.random_search import RandomSearchOptimizer

    # Load dataset (classification example)
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define metrics for classification
    metrics = MetricSet(
        {"accuracy": Metric(name="accuracy", fn=accuracy_score, minimize=False)},
        primary_metric_name="accuracy"
    )
    # Load classification search space
    search_space = SearchSpace.from_yaml("classification_space.yaml")
    # Initialize eZAutoML for classification
    ezautoml = eZAutoML(
        search_space=search_space,
        task=TaskType.CLASSIFICATION,
        metrics=metrics,
        max_trials=10,
        max_time=600,  
        seed=42
    )
    ezautoml.fit(X_train, y_train)
    test_accuracy = ezautoml.test(X_test, y_test)
    ezautoml.summary(k=5)
```

## Contributing

We welcome contributions to eZAutoML! If you'd like to contribute, please fork the repository and submit a pull request with your changes. For detailed information on how to contribute, please refer to our contributing guide.

## License 

eZAutoML is licensed under the BSD 3-Clause License. See the [LICENSE](./LICENSE) file for more information.
