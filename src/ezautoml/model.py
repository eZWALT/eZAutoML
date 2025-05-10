import time
from ezautoml.results.history import History
from ezautoml.evaluation.evaluator import Evaluator
from ezautoml.optimization.optimizers.random_search import RandomSearchOptimizer
from ezautoml.results.trial import Trial
from ezautoml.space.search_space import SearchSpace
from ezautoml.evaluation.metric import MetricSet
from ezautoml.data.loader import DatasetLoader
from ezautoml.evaluation.task import TaskType

from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class eZAutoML:
    def __init__(
        self,
        search_space: SearchSpace,
        task: TaskType,
        metrics: MetricSet,
        optimizer_cls=RandomSearchOptimizer,
        max_trials=100,
        max_time=3600,
        seed=42,
        verbose=True
    ):
        self.verbose = verbose
        self.search_space = search_space
        self.metrics = metrics
        self.optimizer_cls = optimizer_cls
        self.max_trials = max_trials
        self.max_time = max_time
        self.seed = seed
        self.task = task 
        
        assert self.task.value == search_space.task.value

        self.history = History()
        self.evaluator = Evaluator(metric_set=metrics)
        self.fitted_model = None
        self.best_config = None
        self.best_model = None  # To keep track of the best model
        self.console = Console()  # Rich Console for better output

    def fit(self, X, y):
        """Run optimization using Random Search."""
        if self.verbose:
            self.console.print("[bold green]Starting optimization...", style="bold green")

        # Initialize the optimizer (Random Search)
        optimizer = self.optimizer_cls(
            space=self.search_space,
            metrics=self.metrics,
            max_trials=self.max_trials,
            max_time=self.max_time,
            seed=self.seed
        )

        start_time = time.time()
        best_accuracy = float('-inf')  # To track the best accuracy
        best_model_config = None  # To store the configuration of the best model

        # Run optimization loop
        while not optimizer.stop_optimization():
            config = optimizer.ask()
            if config is None:
                break

            # Model instantiation and training
            t0 = time.time()
            model = config.model.instantiate(config.model_params)
            model.fit(X, y)
            duration = time.time() - t0

            # Predictions and evaluation
            preds = model.predict(X)
            evaluation = self.evaluator.evaluate(y, preds)

            # Save evaluation result
            config.result = evaluation.results
            optimizer.tell(config)

            # Record trial
            trial = Trial(
                seed=self.seed,
                model_name=config.model.name,
                optimizer_name=optimizer.__class__.__name__,
                evaluation=evaluation,
                duration=duration
            )
            self.history.add(trial)

            if self.verbose:
                self.console.print(f"[Trial {len(self.history.trials)}] Accuracy={evaluation.results['accuracy']:.4f} in {duration:.2f}s", style="dim")

            # Check if this trial is the best so far
            accuracy = evaluation.results["accuracy"]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_config = config  # Keep track of the configuration of the best model

            # Time budget check
            if time.time() - start_time > self.max_time:
                if self.verbose:
                    self.console.print("[bold red]Time budget exhausted.[/bold red]")
                break

        # Once optimization is done, instantiate the best model based on best configuration
        if best_model_config:
            if self.verbose:
                self.console.print(f"[bold green]Best model:[/bold green] {best_model_config.model.name} with accuracy={best_accuracy:.4f}")
            
            # Instantiate and fit the best model with the best configuration
            self.best_config = best_model_config
            self.best_model = best_model_config.model.instantiate(best_model_config.model_params)
            self.best_model.fit(X, y)
        else:
            if self.verbose:
                self.console.print("[bold red]No valid pipeline found.[/bold red]")

    def predict(self, X):
        """Make predictions using the fitted model."""
        if self.best_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.best_model.predict(X)

    def score(self, X, y):
        """Evaluate the fitted model on the provided data."""
        if self.best_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        preds = self.predict(X)
        return self.metrics.primary().fn(y, preds)

    def test(self, X_test, y_test):
        """Evaluate the best model on the test set."""
        if self.best_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Make predictions and evaluate
        predictions = self.predict(X_test)
        test_accuracy = accuracy_score(y_test, predictions)
        self.console.print(f"[bold blue]Test Accuracy:[/bold blue] {test_accuracy:.4f}")
        return test_accuracy

    def summary(self, k=5):
        """Show a summary of the best trials."""
        return self.history.summary(k=k, metrics=[self.metrics.primary_metric_name])

# --- Main Function ---
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from ezautoml.space.component import Component, Tag
from ezautoml.space.hyperparam import Hyperparam, Integer, Real
from ezautoml.space.search_space import SearchSpace
from ezautoml.evaluation.metric import Metric, MetricSet
from ezautoml.optimization.optimizers.random_search import RandomSearchOptimizer
from ezautoml.data.loader import DatasetLoader

def main():
   # --- Load dataset ---
    loader = DatasetLoader(
        local_path="../../data", 
        metadata_path="../../data/metadata.json"
    )
    datasets = loader.load_selected_datasets(groups=["local", "builtin", "torchvision"])
    X, y = datasets["breast_cancer"]  # Replace with any available dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Define metrics and evaluator
    metrics = MetricSet(
        {"accuracy": Metric(name="accuracy", fn=accuracy_score, minimize=False)},
        primary_metric_name="accuracy"
    )
    
    search_space = SearchSpace.from_yaml("classification_space.yaml")

    # Initialize eZAutoML
    ezautoml = eZAutoML(
        search_space=search_space,
        task=TaskType.CLASSIFICATION,
        metrics=metrics,
        max_trials=5,
        max_time=600,  # 10 minutes
        seed=42
    )

    # Fit model
    ezautoml.fit(X_train, y_train)
    # Test using the test data
    test_accuracy = ezautoml.test(X_test, y_test)
    # Show best trial summary
    ezautoml.summary(k=5)

if __name__ == "__main__":
    main()
