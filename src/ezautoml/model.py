# ezautoml/api.py

import time
from ezautoml.results.history import History
from ezautoml.evaluation.evaluator import Evaluator
from ezautoml.space.search_space import SearchSpace
from ezautoml.optimization.optimizers.random_search import RandomSearchOptimizer 
from ezautoml.results.trial import Trial
from ezautoml.evaluation.metric import MetricSet, Metric
from ezautoml.evaluation.task import TaskType

class eZAutoML:
    def __init__(
        self,
        search_space: SearchSpace,
        metrics: MetricSet,
        optimizer_cls=RandomSearchOptimizer,
        max_trials=100,
        max_time=3600,
        seed=42,
    ):
        self.search_space = search_space
        self.metrics = metrics
        self.optimizer_cls = optimizer_cls
        self.max_trials = max_trials
        self.max_time = max_time
        self.seed = seed

        self.history = History()
        self.evaluator = Evaluator(metric_set=metrics)
        self.fitted_model = None
        self.best_config = None

    def fit(self, X, y):
        print("[eZAutoML] Starting optimization...")
        optimizer = self.optimizer_cls(
            space=self.search_space,
            metrics=self.metrics,
            max_trials=self.max_trials,
            max_time=self.max_time,
            seed=self.seed
        )

        start_time = time.time()

        while not optimizer.is_converged():
            config = optimizer.ask()
            if config is None:
                break

            t0 = time.time()
            model = config.model.instantiate(config.model_params)
            model.fit(X, y)
            duration = time.time() - t0

            preds = model.predict(X)
            evaluation = self.evaluator.evaluate(y, preds)

            config.result = evaluation.results
            optimizer.tell(config)

            trial = Trial(
                seed=self.seed,
                model_name=config.model.name,
                optimizer_name=optimizer.__class__.__name__,
                evaluation=evaluation,
                duration=duration
            )
            self.history.add(trial)

            print(f"[Trial {len(self.history)}] Accuracy={evaluation.results['accuracy']:.4f} in {duration:.2f}s")

            if time.time() - start_time > self.max_time:
                print("[eZAutoML] Time budget exhausted.")
                break

        best_trial = self.history.get_best()
        if best_trial:
            print(f"[eZAutoML] Best model: {best_trial.model_name} with {best_trial.evaluation.primary_score:.4f}")
            self.best_config = best_trial
            self.fitted_model = best_trial.config.model.instantiate(best_trial.config.model_params)
            self.fitted_model.fit(X, y)
        else:
            print("[eZAutoML] No valid pipeline found.")

    def predict(self, X):
        if self.fitted_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.fitted_model.predict(X)

    def score(self, X, y):
        if self.fitted_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        preds = self.predict(X)
        return self.metrics.primary().fn(y, preds)

    def summary(self, k=5):
        return self.history.summary(k=k, metrics=[self.metrics.primary().name])
