from abc import ABC, abstractmethod
from typing import List, Optional, Union
import random
import time

from ezautoml.space.search_space import SearchSpace
from ezautoml.space.search_point import SearchPoint
from ezautoml.evaluation.metric import MetricSet

class Optimizer(ABC):
    """Abstract base optimizer for CASH: model selection + hyperparameter optimization.

    Manages trial sampling, tracking, and stopping based on trial count or time.
    """

    def __init__(
        self,
        metrics: MetricSet,
        space: SearchSpace,
        max_trials: int,
        max_time: int,  # in seconds
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        self.metrics = metrics
        self.space = space
        self.max_trials = max_trials
        self.max_time = max_time
        self.seed = seed
        self.rng = random.Random(seed)
        self.verbose = verbose

        self.trial_count = 0
        self.trials: List[SearchPoint] = []
        self.start_time = time.time()

    @abstractmethod
    def tell(self, report: SearchPoint) -> None:
        """Update the optimizer with the result of a completed trial."""
        pass

    @abstractmethod
    def ask(self, n: int = 1) -> Union[SearchPoint, List[SearchPoint]]:
        """Request one or more new configurations to evaluate."""
        pass

    def get_trials(self) -> List[SearchPoint]:
        """Return all completed trials."""
        return self.trials

    def has_reached_max_trials(self) -> bool:
        return self.trial_count >= self.max_trials

    def has_reached_max_time(self) -> bool:
        return (time.time() - self.start_time) >= self.max_time

    def stop_optimization(self) -> bool:
        """Check stopping condition based on trial count or time."""
        return self.has_reached_max_trials() or self.has_reached_max_time()

    def get_best_trial(self) -> Optional[SearchPoint]:
        """Return the best trial based on the main metric (first in the set)."""
        if not self.trials:
            return None

        main_metric = self.metrics.primary
        key = main_metric.name
        reverse = not main_metric.minimize

        return max(
            self.trials,
            key=lambda t: t.result.get(key, float("-inf") if reverse else float("inf")),
        )

    @classmethod
    def create(
        cls,
        space: SearchSpace,
        metrics: MetricSet,
        max_trials: int,
        max_time: int,
        seed: Optional[int] = None,
    ) -> 'Optimizer':
        return cls(
            metrics=metrics,
            space=space,
            max_trials=max_trials,
            max_time=max_time,
            seed=seed,
        )
