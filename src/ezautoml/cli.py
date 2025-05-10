# eZAutoML/cli.py
import argparse
from loguru import logger 

from ezautoml.__version__ import __version__

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
        help="Path to the dataset file (CSV, Parquet...)"
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

    from eZAutoML.src.ezautoml.runner import AutoMLRunner
    from ezautoml.evaluation.evaluator import Evaluator
    from ezautoml.optimization.optimizers.random_search import RandomSearchOptimizer
    from ezautoml.space.search_space import SearchSpace
    from loguru import logger
    
    search_space = SearchSpace.from_file("search_space.yaml")
    optimizer = RandomSearchOptimizer()
    evaluator = Evaluator(X_train, y_train, X_test, y_test)

    runner = AutoMLRunner(optimizer=optimizer, evaluator=evaluator)
    runner.run(n_trials=10)

    best = runner.get_best()
    print(f"Best trial accuracy: {best.result.get('accuracy')}")

        
if __name__ == "__main__":
    run_cli()
