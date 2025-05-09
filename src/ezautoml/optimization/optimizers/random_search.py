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

    def tell(self, report: SearchPoint) -> None:
        """Record the result of a completed trial."""
        logger.info(f"[TELL] Received report:\n{report}")
        self.trials.append(report)
        self.trial_count += 1

    def ask(self, n: int = 1) -> Union[SearchPoint, List[SearchPoint]]:
        """Sample new candidate configurations, unless max trials or time exceeded. m"""
        if self.stop_optimization():
            logger.info("Stopping condition met (max trials or time).")
            return []

        trials = [self.space.sample() for _ in range(n)]
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

# This is a minimal example of using random search once
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from ezautoml.space.component import Component, Tag
    from ezautoml.space.hyperparam import Hyperparam, Integer, Real
    from ezautoml.space.search_space import SearchSpace
    from ezautoml.evaluation.metric import Metric, MetricSet
    from ezautoml.evaluation.task import TaskType
    from ezautoml.optimization.optimizer import Optimizer
    from loguru import logger

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define performance metrics
    metrics = MetricSet(
        {"accuracy": Metric(name="accuracy", fn=accuracy_score, minimize=False)},
        primary_metric_name="accuracy"
    )

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

    # Define components
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

    scaler_component = Component("StandardScaler", StandardScaler, [])
    pca_component = Component("PCA", PCA, [Hyperparam("n_components", Real(0.5, 0.99))])

    # Create search space with model and preprocessing steps
    search_space = SearchSpace(
        models=[rf_component, dt_component, lr_component],
        data_processors=[scaler_component],
        feature_processors=[pca_component],
        task=TaskType.CLASSIFICATION
    )

    # Create optimizer with maximum trials and time limit
    optimizer = RandomSearchOptimizer.create(
        space=search_space,
        metrics=metrics,
        seed=42,
        max_trials=20,  # Limit the number of trials
        max_time=3600,  # Limit the total time (1 hour)
    )

    # Sample a trial
    trial = optimizer.ask()
    logger.success(f"[TRIAL] Sampled configuration:\n{trial}")

    # Fit the model using the trial configuration
    # Assuming the trial result contains actual model instantiation
    model = trial.model.instantiate(trial.model_params)
    scaler = trial.data_processors[0].instantiate(trial.data_params_list[0])
    pca = trial.feature_processors[0].instantiate(trial.feature_params_list[0])

    # Preprocess the data: Scaling and PCA
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_pca = pca.fit_transform(X_train_scaled)
    model.fit(X_train_pca, y_train)

    # Evaluate the model
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    predictions = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, predictions)

    # Report the trial's result
    trial.result = {"accuracy": accuracy}
    optimizer.tell(trial)

    # Get the best trial after a few iterations
    best_trial = optimizer.get_best_trial()
    logger.success(f"Best trial: {best_trial}")

    # Print final results
    if best_trial:
        logger.success(f"Best trial configuration: {best_trial.describe()}")
        logger.success(f"Best trial accuracy: {best_trial.result.get('accuracy')}")