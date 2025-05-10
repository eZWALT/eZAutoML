import time
import optuna
from ezautoml.space.search_space import SearchSpace
from ezautoml.space.search_point import SearchPoint
from ezautoml.optimization.optimizer import Optimizer
from ezautoml.evaluation.metric import MetricSet, Metric, Comparison
from ezautoml.parsers.optuna import OptunaParser
from ezautoml.results.trial import Trial
from sklearn.model_selection import train_test_split
from loguru import logger
from typing import Optional, Union, List
import numpy as np





class OptunaOptimizer(Optimizer):
    """Optuna optimization strategy for CASH (model selection + hyperparameter tuning)."""

    def __init__(
        self,
        metrics: MetricSet,
        space: SearchSpace,
        X: np.ndarray,
        y: np.ndarray,
        max_trials: int,
        max_time: int,
        seed: Optional[int] = 42,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            metrics=metrics,
            space=space,
            max_trials=max_trials,
            max_time=max_time,
            seed=seed,
        )

        self.X = X
        self.y = y
        self.verbose = verbose

        # Define the optimization direction
        primary_metric = metrics.primary
        direction = "maximize" if not primary_metric.minimize else "minimize"

        # Create an Optuna study for optimization
        self.study = optuna.create_study(direction=direction)
        self.trial_count = 0

        # Initialize the OptunaParser
        self.parser = OptunaParser(space)

    def tell(self, report: SearchPoint) -> None:
        """Record the result of a completed trial."""
        if self.verbose:
            logger.info(f"[TELL] Received report: {report.result}")
            print(report)

        # Extract the result from the report, assuming `result` contains an `Evaluation` object
        if report.result:
            score = report.result.results["accuracy"]  # Assuming 'accuracy' is one of the keys in the results

            # Check if any trials are completed
            if len(self.study.trials) == 0 or all(trial.state != optuna.trial.TrialState.COMPLETE for trial in self.study.trials):
                # Add the first trial (initial trial) if no trials are completed yet
                trial = optuna.trial.FrozenTrial(
                    number=self.trial_count,
                    value=score,
                    state=optuna.trial.TrialState.COMPLETE,
                    params=report.model_params,  # use model_params for the trial
                )
                self.study.add_trial(trial)
                if self.verbose:
                    logger.info(f"First trial added with score {score}.")
            else:
                # Check if the best trial has been completed
                if self.study.best_trial.state == optuna.trial.TrialState.COMPLETE:
                    comparison = self.metrics.primary.is_improvement(self.study.best_value, score)

                    # If new trial is better, add it
                    if comparison == Comparison.BETTER:
                        trial = optuna.trial.FrozenTrial(
                            number=self.trial_count,
                            value=score,
                            state=optuna.trial.TrialState.COMPLETE,
                            params=report.model_params,  # use model_params for the trial
                        )
                        self.study.add_trial(trial)

                        if self.verbose:
                            logger.info(f"New best trial found with score {score}.")
                    else:
                        if self.verbose:
                            logger.info(f"Trial did not improve. Current best: {self.study.best_value}")
                else:
                    # If no trials are fully completed, just add the current trial
                    trial = optuna.trial.FrozenTrial(
                        number=self.trial_count,
                        value=score,
                        state=optuna.trial.TrialState.COMPLETE,
                        params=report.model_params,  # use model_params for the trial
                    )
                    self.study.add_trial(trial)

                    if self.verbose:
                        logger.info(f"First completed trial added with score {score}.")

            self.trial_count += 1


    def ask(self, n: int = 1) -> Union[SearchPoint, List[SearchPoint]]:
        """Sample new candidate configurations from the search space."""
        if self.stop_optimization():
            if self.verbose:
                logger.info("Stopping condition met (max trials or time).")
            return []

        trials = []
        for _ in range(n):
            trial = self.study.ask()

            # Convert the trial into a full configuration
            config = self.parser.convert_to_search_point(trial)

            # Here, we use the instantiate method on the Component instance, not the model class
            model = config.model.instantiate(config.model_params)

            # TODO APPLY DATA PROCESSORS

            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.3, random_state=self.seed)

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_val)
            evaluation_score = self.metrics.primary.evaluate(y_val, y_pred)

            # Attach the evaluation result to the configuration
            config.result = {self.metrics.primary.name: evaluation_score}

            # Append the configuration to the trial list
            trials.append(config)

            if self.verbose:
                logger.info(f"[ASK] Sampled configuration: {config} - Evaluation: {evaluation_score}")

        return trials if n > 1 else trials[0]

    def get_best_trial(self) -> Optional[SearchPoint]:
        """Return the best trial based on the primary metric."""
        if not self.study.trials:
            return None

        best_trial = self.study.best_trial
        return self.parser.convert_to_search_point(best_trial)

    def objective(self, trial: optuna.Trial):
        """Define the optimization objective."""
        config = self.parser.convert_to_search_point(trial)

        # Apply data processors, feature processors, etc., if necessary
        model = config.model.instantiate(config.model_params)

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.3, random_state=self.seed)

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_val)
        evaluation_score = self.metrics.primary.evaluate(y_val, y_pred)

        return evaluation_score

    def optimize(self):
        """Run the optimization using the Optuna study."""
        self.study.optimize(self.objective, n_trials=self.max_trials)

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

    # 1. Initialize DatasetLoader
    loader = DatasetLoader(local_path="../../data", metadata_path="../../data/metadata.json")
    datasets = loader.load_selected_datasets(groups=["local", "builtin", "torchvision"])  # Load datasets

    # 2. Select a dataset (e.g., breast cancer dataset)
    X, y = datasets["breast_cancer"]  # Adjust depending on the dataset you want to use

    # 3. Split the dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 4. Define metrics and evaluator
    metrics = MetricSet(
        {"accuracy": Metric(name="accuracy", fn=accuracy_score, minimize=False)},
        primary_metric_name="accuracy"
    )
    evaluator = Evaluator(metric_set=metrics)

    # 5. Define hyperparameters for models
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

    # 6. Define model components
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

    # 7. Define search space and optimizer
    search_space = SearchSpace(
        models=[rf_component, dt_component, lr_component],
        task="classification"
    )

    optimizer = OptunaOptimizer(
        X=X,
        y=y,
        space=search_space,
        metrics=metrics,
        max_trials=10,
        max_time=3600,
        seed=42,
        verbose=True
    )

    # 8. Initialize trial history
    history = History()

    # 9. Run trials
    for _ in range(100):
        trial_config = optimizer.ask()
        if not trial_config:
            break

        start = time.time()

        # 10. Instantiate model and fit
        model = trial_config.model.instantiate(trial_config.model_params)
        model.fit(X_train, y_train)

        # 11. Make predictions and evaluate
        predictions = model.predict(X_test)
        evaluation = evaluator.evaluate(y_test, predictions)
        duration = time.time() - start

        # 12. Update optimizer with trial results
        trial_config.result = evaluation.results
        optimizer.tell(trial_config)

        # 13. Record trial
        trial = Trial(
            seed=42,
            model_name=trial_config.model.name,
            optimizer_name="RandomSearch",
            evaluation=evaluation,
            duration=duration
        )
        history.add(trial)

    # 14. Print summary of the best trials
    history.summary(k=50, metrics=["accuracy"])
