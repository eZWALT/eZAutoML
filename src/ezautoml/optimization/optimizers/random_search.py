from typing import List, Optional, Union
from loguru import logger

from ezautoml.space.search_space import SearchSpace
from ezautoml.space.search_point import SearchPoint
from ezautoml.optimization.optimizer import Optimizer
from ezautoml.evaluation.task import TaskType

# TODO TRIAL

class RandomSearchOptimizer(Optimizer):
    """A simple random search optimizer."""

    def tell(self, report: SearchPoint) -> None:
        logger.info(f"[TELL] Received report for trial:\n{report}")

    def ask(self, n: int = 1) -> Union[SearchPoint, List[SearchPoint]]:
        trials = [self.space.sample() for _ in range(n)]
        logger.info(f"[ASK] Sampling {n} trial(s)")
        return trials if n > 1 else trials[0]


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from ezautoml.space.component import Component, Tag
    from ezautoml.space.hyperparam import Hyperparam, Integer
    from ezautoml.space.search_space import SearchSpace
    from ezautoml.evaluation.metric import Metric, MetricSet
    from loguru import logger

    metrics = MetricSet([
        Metric(name="accuracy", fn=accuracy_score, minimize=False)
    ])

    rf_params = [
        Hyperparam("n_estimators", Integer(10, 100)),
        Hyperparam("max_depth", Integer(3, 15)),
    ]
    dt_params = [
        Hyperparam("max_features", Integer(10, 100)),
        Hyperparam("max_depth", Integer(1, 100)),
    ]

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
    

    search_space = SearchSpace(
        models=[rf_component,dt_component],
        data_processors=[],
        feature_processors=[],
        task=TaskType.CLASSIFICATION
    )

    optimizer = RandomSearchOptimizer.create(
        space=search_space,
        metrics=metrics,
        seed=42
    )

    trial = optimizer.ask()
    logger.success(f"[TRIAL] Sampled configuration:\n{trial}")

    # Simulate fake evaluation
    trial.set_result({"accuracy": 0.85})
    optimizer.tell(trial)
