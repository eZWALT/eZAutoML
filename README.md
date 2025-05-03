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
  <img src="./resources/logo_white.jpeg" alt="eZAutoML Logo" width="300"/>
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

Usage:

```bash
ezautoml --dataset <path_to_data> --target <target_name> --task <classification|regression> --models <model1,model2,...> --cv <folds> --output <path_to_output>
```

Options:
- dataset: Path to the dataset file (CSV, parquet...)
- target: The target column name for prediction
- task: Task type: classification or regression
- search: Black-box optimization algorithm to perform
- models: Comma-separated list of models to use (e.g., lr,rf,xgb). Use initials!
- cv: Number of cross-validation folds (if needed)
- output: Directory to save the output models/results
- trials: Maximum number of trials inside an optimiation algorithm
- preprocess: Whether to perform minimal preprocessing (Scaling, Encoding...) or not
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
???
```


## WIP

## WIP TODO List for eZAutoML

### 1. **Core System Setup**
- [ ] **Implement Dataset Loading (`datasets.py`)**
   - Build a utility to load datasets from various formats (CSV, Parquet, etc.).
   - Implement functionality to split datasets into train and test sets.

- [ ] **Preprocessing (`preprocess.py`)**
   - Implement basic preprocessing such as:
     - Feature scaling (StandardScaler)
     - Label encoding for classification tasks
     - Handling missing values (if necessary)
   - **Optional**: Extend to more advanced preprocessing in the future.
  
### 2. **Model Implementation**
- [ ] **Model Definitions (`models.py`)**
   - Implement a list of models:
     - SVM, RandomForest, XGBoost, etc.
     - Ensure models can be easily swapped based on the user's request in CLI (`--models` flag).
   
- [ ] **Search Strategy (`search.py`)**
   - Implement the abstract optimizer class, and separate search strategies such as:
     - **Random Search**: Use for hyperparameter tuning.
     - **Grid Search**: For exhaustive search of hyperparameters.
   - Provide flexibility to add new strategies later.

### 3. **Model Evaluation**
- [x] **Evaluator (`evaluation.py`)**
   - Implement cross-validation to assess model performance.
   - Support various metrics (accuracy, F1 score, etc.) based on the task (classification/regression).

- [ ] **Leaderboard (`reporting.py`)**
   - Track and store model performance (accuracy, metrics).
   - Build a leaderboard that ranks models based on their cross-validation score.

### 4. **Optimization System**
- [ ] **Abstract Optimizer (`search.py`)**
   - Implement a base class for optimizers, handling setup and execution of hyperparameter search.
   - Design the optimizer to integrate with different search strategies (Random Search, Grid Search).
  
- [ ] **Random Search Optimizer** 
   - Implement random hyperparameter search strategy.
   - Randomly sample hyperparameters from predefined search spaces.
   - Use the evaluator to assess performance during each trial.

### 5. **History Tracking**
- [ ] **Build History Logging System (`history.py`)**
   - Implement a system to store trial results (model parameters, validation scores, etc.).
   - Provide an easy way to retrieve and analyze previous experiment results.
  
### 6. **Reporting and Output**
- [ ] **Reporting (`reporting.py`)**
   - Create functionality to log experiment results.
   - Optionally generate visualization such as bar plots for leaderboard.
   - Save reports and models to the specified output directory.
  
### 7. **Configuration System**
- [ ] **Config Management (`config.py`)**
   - Define default search spaces for hyperparameters.
   - Allow easy configuration of model hyperparameters and search spaces.
   - Ensure flexibility for future extension.





## Contributing

We welcome contributions to eZAutoML! If you'd like to contribute, please fork the repository and submit a pull request with your changes. For detailed information on how to contribute, please refer to our contributing guide.

## License 

eZAutoML is licensed under the BSD 3-Clause License. See the [LICENSE](./LICENSE) file for more information.
