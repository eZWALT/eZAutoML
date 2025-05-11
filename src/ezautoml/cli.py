import argparse
import os
from loguru import logger
from ezautoml.results.history import History
from ezautoml.evaluation.evaluator import Evaluator
from ezautoml.optimization.optimizers.random_search import RandomSearchOptimizer
from ezautoml.results.trial import Trial
from ezautoml.space.search_space import SearchSpace
from ezautoml.evaluation.metric import MetricSet
from ezautoml.data.loader import DatasetLoader
from ezautoml.evaluation.task import TaskType
from ezautoml.model.ezautoml import eZAutoML
from rich.console import Console
from rich.table import Table

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(
        prog="ezautoml",
        description="A Democratized, lightweight and modern framework for Python Automated Machine Learning.",
        epilog="For more info, visit: https://github.com/eZWALT/eZAutoML"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset file (CSV) or a list of paths"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="The target column name for prediction"
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        required=True,
        help="Task type: classification or regression"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="lgbm,xgb,rf",
        help="Comma-separated list of models to use (e.g., lr,rf,xgb). Use initials!"
    )
    parser.add_argument(
        "--search",
        choices=["random", "optuna"],
        default="random",
        help="Black-box optimization algorithm to perform"
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (if needed)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="accuracy,f1_score",
        help="Comma-separated list of metrics to use (e.g., accuracy,f1_score for classification or mse,r2 for regression)"
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="accuracy",
        help="Scoring metric to use for evaluation"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Maximum number of trials inside an optimization algorithm"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Directory to save the output models/results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"eZAutoML {__version__}",
        help="Show the current version"
    )

    return parser.parse_args()

def run_cli():
    args = parse_args()

    # Initialize DatasetLoader
    loader = DatasetLoader(local_path="../../data", metadata_path="../../data/metadata.json")
    
    # Handle single or multiple dataset paths
    dataset_paths = args.dataset.split(",")  # Split if there are multiple paths provided
    
    # Load user datasets using the DatasetLoader's load_user_datasets method
    logger.info(f"Loading datasets from: {dataset_paths}")
    datasets = loader.load_user_datasets(file_paths=dataset_paths, metadata={args.dataset: args.target})
    
    # Assuming only one dataset is loaded; you can adapt it for multiple datasets if needed
    if len(datasets) == 0:
        logger.error("No datasets loaded.")
        return
    
    # For simplicity, we take the first loaded dataset
    dataset_name = list(datasets.keys())[0]
    X, y = datasets[dataset_name]

    # Split dataset into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define metrics and evaluator
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    if args.task == "classification":
        metrics = MetricSet(
            {"accuracy": accuracy_score, "f1_score": f1_score},
            primary_metric_name="accuracy"
        )
        task_type = TaskType.CLASSIFICATION
    else:
        metrics = MetricSet(
            {"mse": mean_squared_error, "r2": r2_score},
            primary_metric_name="r2"
        )
        task_type = TaskType.REGRESSION

    # Define search space (You might want to load this from a file)
    search_space = SearchSpace.from_file("search_space.yaml")

    # Initialize eZAutoML
    ezautoml = eZAutoML(
        search_space=search_space,
        task=task_type,
        metrics=metrics,
        max_trials=args.trials,
        max_time=600,  # 10 minutes
        seed=42,
        verbose=args.verbose
    )

    # Fit model
    ezautoml.fit(X_train, y_train)
    
    # Test using the test data
    test_accuracy = ezautoml.test(X_test, y_test)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    # Show best trial summary
    summary = ezautoml.summary(k=5)
    logger.info(f"Best Trials Summary:\n{summary}")

if __name__ == "__main__":
    run_cli()
