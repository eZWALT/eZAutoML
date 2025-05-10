from ezautoml.space.search_space import SearchSpace
from ezautoml.space.search_point import SearchPoint
from ezautoml.optimization.optimizer import Optimizer
from ezautoml.evaluation.metric import MetricSet
from loguru import logger
from typing import List, Optional, Union
import time

class RandomSearchOptimizer(Optimizer):
    """Random search strategy for CASH (model selection + hyperparameter tuning)."""

    def __init__(
        self,
        metrics: MetricSet,
        space: SearchSpace,
        max_trials: int,
        max_time: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            metrics=metrics,
            space=space,
            max_trials=max_trials,
            max_time=max_time,
            seed=seed,
        )
        __name__ = "RandomSearchOptimizer"

    def tell(self, report: SearchPoint) -> None:
        """Record the result of a completed trial."""
        if self.verbose:
            logger.info(f"[TELL] Received report:\n{report}")
        self.trials.append(report)
        self.trial_count += 1

    def ask(self, n: int = 1) -> Union[SearchPoint, List[SearchPoint]]:
        """Sample new candidate configurations, unless max trials or time exceeded. m"""
        if self.stop_optimization():
            if self.verbose:
                logger.info("Stopping condition met (max trials or time).")
            return []

        trials = [self.space.sample() for _ in range(n)]
        if self.verbose:
            logger.info(f"[ASK] Sampling {n} configuration(s).")
        return trials if n > 1 else trials[0]

    def get_best_trial(self) -> Optional[SearchPoint]:
        """Return the best trial based on the primary metric."""
        if not self.trials:
            return None

        main_metric = self.metrics.primary
        key = main_metric.name
        reverse = not main_metric.minimize

        # Return the best trial based on the primary metric
        return max(
            self.trials,
            key=lambda t: t.result.get(key, float("-inf") if reverse else float("inf")),
        )


if __name__ == "__main__":
    import time
    import random
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris, load_breast_cancer
    from sklearn.metrics import accuracy_score

    from ezautoml.space.component import Component, Tag
    from ezautoml.space.hyperparam import Hyperparam, Integer, Real
    from ezautoml.space.search_space import SearchSpace
    from ezautoml.evaluation.metric import Metric, MetricSet
    from ezautoml.evaluation.evaluator import Evaluator
    from ezautoml.evaluation.task import TaskType
    from ezautoml.results.trial import Trial
    from ezautoml.results.history import History
    
    from ezautoml.data.loader import DatasetLoader

    # Initialize DatasetLoader
    loader = DatasetLoader(local_path="../../data", metadata_path="../../data/metadata.json")
    datasets = loader.load_selected_datasets(groups=["local", "builtin", "torchvision"])  # Load datasets

    # Select a dataset (for example, load the breast cancer dataset)
    X, y = datasets["breast_cancer"]  # Adjust depending on the dataset you want to use

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define metrics and evaluator
    metrics = MetricSet(
        {"accuracy": Metric(name="accuracy", fn=accuracy_score, minimize=False)},
        primary_metric_name="accuracy"
    )
    evaluator = Evaluator(metric_set=metrics)

    # Define hyperparameters for models
    rf_params = [
        Hyperparam("n_estimators", Integer(10, 100)),
        Hyperparam("max_depth", Integer(3, 15)),
    ]
    dt_params = [
        Hyperparam("max_features", Integer(10, 100)),
        Hyperparam("max_depth", Integer(1, 100)),
    ]
    lr_params = [
        Hyperparam("C", Real(0.01, 10)),
        Hyperparam("max_iter", Integer(50, 500)),
    ]

    # Define model components
    rf_component = Component(
        name="RandomForest",
        tag=Tag.MODEL_SELECTION,
        constructor=RandomForestClassifier,
        hyperparams=rf_params,
    )
    dt_component = Component(
        name="DecisionTree",
        tag=Tag.MODEL_SELECTION,
        constructor=DecisionTreeClassifier,
        hyperparams=dt_params,
    )
    lr_component = Component(
        name="LogisticRegression",
        tag=Tag.MODEL_SELECTION,
        constructor=LogisticRegression,
        hyperparams=lr_params,
    )

    # Define search space and optimizer
    search_space = SearchSpace(
        models=[rf_component, dt_component, lr_component],
        task="classification"
    )

    optimizer = RandomSearchOptimizer(
        space=search_space,
        metrics=metrics,
        max_trials=100,
        max_time=3600,
        seed=42
    )

    # Initialize trial history
    history = History()

    # Run trials
    for _ in range(100):
        trial_config = optimizer.ask()
        if not trial_config:
            break

        start = time.time()

        # Instantiate model and fit
        model = trial_config.model.instantiate(trial_config.model_params)
        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_test)
        evaluation = evaluator.evaluate(y_test, predictions)
        duration = time.time() - start

        # Update optimizer with trial results
        trial_config.result = evaluation.results
        optimizer.tell(trial_config)

        # Record trial
        trial = Trial(
            seed=42,
            model_name=trial_config.model.name,
            optimizer_name="RandomSearch",
            evaluation=evaluation,
            duration=duration
        )
        history.add(trial)

    # Print summary of the best trials
    history.summary(k=50, metrics=["accuracy"])