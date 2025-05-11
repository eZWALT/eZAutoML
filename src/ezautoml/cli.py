import argparse
import os
import pandas as pd
import re
from loguru import logger

from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from rich.console import Console
from rich.table import Table

from ezautoml.data.preprocess import prepare_data
from ezautoml.optimization.optimizers.random_search import RandomSearchOptimizer
from ezautoml.optimization.optimizers.optuna import OptunaOptimizer
from ezautoml.space.search_space import SearchSpace
from ezautoml.evaluation.metric import MetricSet, Metric
from ezautoml.evaluation.task import TaskType
from ezautoml.model import eZAutoML
from ezautoml.__version__ import __version__

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    """
    Parses the command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="ezautoml",
        description="A Democratized, lightweight and modern framework for Python Automated Machine Learning.",
        epilog="For more info, visit: https://github.com/eZWALT/eZAutoML"
    )

    # Required arguments
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file (CSV)")
    parser.add_argument("--target", type=str, required=True, help="The target column name for prediction")
    parser.add_argument(
        "--task", 
        choices=["classification", "regression", "c", "r"], 
        required=True, 
        help="Task type: 'classification', 'regression', 'c' for classification, or 'r' for regression"
    )
    # Optional arguments
    parser.add_argument("--models", type=str, default="lgbm,xgb,rf", help="Comma-separated list of models to use (e.g., lr,rf,xgb). Use initials!")
    parser.add_argument("--search", choices=["random", "optuna"], default="random", help="Optimization algorithm to perform")
    parser.add_argument("--trials", type=int, default=10, help="Maximum number of trials inside an optimization algorithm")
    parser.add_argument("--output", type=str, default=".", help="Directory to save the output models/results")
    parser.add_argument("--save", action="store_true", help="Directory to save the output models/results")
    parser.add_argument("--verbose", action="store_true", help="Increase logging verbosity")
    parser.add_argument("--version", action="version", version=f"eZAutoML {__version__}", help="Show the current version")

    return parser.parse_args()

def sanitize_feature_names(df):
    """
    Sanitizes column names by replacing non-alphanumeric characters with underscores.
    """
    sanitized_columns = [re.sub(r'[^0-9a-zA-Z_]', '_', col) for col in df.columns]
    df.columns = sanitized_columns
    return df

def load_and_prepare_data(dataset_path, target_column):
    """
    Loads and preprocesses the dataset.
    It sanitizes feature names, handles missing values, encodes categorical variables,
    and scales numerical features.
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist.")
        return None, None

    data = pd.read_csv(dataset_path)
    
    # Sanitize column names
    data = sanitize_feature_names(data)

    # Prepare the data using the prepare_data function
    X, y = prepare_data(data, target_column)

    if X is None or y is None:
        logger.error(f"Failed to prepare the data for model training.")
        return None, None

    return X, y

def get_task_type_and_metrics(task):
    """
    Returns the appropriate TaskType and metrics based on the task (classification or regression).
    """
    # Map short options 'c' and 'r' to their corresponding full task names
    task_mapping = {
        "c": "classification",
        "r": "regression",
        "classification": "classification",
        "regression": "regression"
    }

    task = task_mapping.get(task, task)  # Get the corresponding full task name

    if task == "classification":
        task_type = TaskType.CLASSIFICATION
        metrics = MetricSet(
            {
                "accuracy": Metric(name="accuracy", fn=accuracy_score, minimize=False),
                "f1_score": Metric(name="f1_score", fn=f1_score, minimize=False, default_kwargs={"average": "macro"})
            },
            primary_metric_name="f1_score"
        )
    elif task == "regression":
        task_type = TaskType.REGRESSION
        metrics = MetricSet(
            {
                "mse": Metric(name="mse", fn=mean_squared_error, minimize=True),
                "r2": Metric(name="r2", fn=r2_score, minimize=False)
            },
            primary_metric_name="mse"
        )

    return task_type, metrics

def select_optimizer(search_strategy):
    """
    Returns the optimizer class based on the search strategy.
    Displays a WIP message for Optuna and stops the execution.
    """
    if search_strategy == "optuna":
        log_wip_message()  # Log the WIP message
        raise SystemExit("Optuna optimizer is currently a Work In Progress. The process will be terminated.")  # Terminate the process
    else:
        optimizer_cls = RandomSearchOptimizer

    return optimizer_cls

def log_wip_message():
    """
    Logs a Work In Progress (WIP) message for the Optuna optimizer.
    """
    logger.warning("Optuna optimizer is WIP (Work in Progress). Please proceed with caution.")

def save_results(ezautoml, output_dir):
    """
    Saves the results and models to the output directory.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ezautoml.history.to_json(os.path.join(output_dir, "history.json"))
        ezautoml.history.to_csv(os.path.join(output_dir, "history_summary.csv"))

def main():
    """
    Main function to orchestrate the workflow.
    """
    # Parse arguments
    args = parse_args()

    # Load and prepare dataset
    X, y = load_and_prepare_data(args.dataset, args.target)
    if X is None or y is None:
        return

    task_mapping = {
        "c": "classification",
        "r": "regression",
        "classification": "classification",
        "regression": "regression"
    }

    # Normalize task
    task = task_mapping.get(args.task, args.task)

    # Get task type and metrics
    task_type, metrics = get_task_type_and_metrics(task)

    # Load search space from YAML or a predefined space (based on task)
    search_space_file = "classification_space.yaml" if task == "classification" else "regression_space.yaml"
    search_space = SearchSpace.from_yaml(search_space_file)

    # Select optimizer based on the search argument
    optimizer_cls = select_optimizer(args.search)

    # Instantiate eZAutoML
    ezautoml = eZAutoML(
        search_space=search_space,
        task=task_type,
        metrics=metrics,
        optimizer_cls=optimizer_cls,
        max_trials=args.trials,
        verbose=args.verbose
    )

    # Fit model
    ezautoml.fit(X, y)

    # Test using the test data
    test_accuracy = ezautoml.test(X, y)
    logger.info(f"Test accuracy: {test_accuracy}")

    # Show summary of best trials
    summary = ezautoml.summary(k=10)

    # Save results to the output directory
    if args.save:
        save_results(ezautoml, args.output)

if __name__ == "__main__":
    main()
