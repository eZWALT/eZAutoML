from abc import ABC, abstractmethod
from typing import List, Optional, Union
import random

from loguru import logger

from ezautoml.space.search_space import SearchSpace
from ezautoml.space.search_point import SearchPoint
from ezautoml.evaluation.metric import MetricSet


class Optimizer(ABC):
    """Abstract base optimizer that works with ezautoml's SearchSpace and SearchPoint."""

    def __init__(
        self,
        metrics: MetricSet,
        space: SearchSpace,
        seed: Optional[int] = None,
    ) -> None:
        self.metrics = metrics
        self.space = space
        self.seed = seed
        self.rng = random.Random(seed)

    @abstractmethod
    def tell(self, report: SearchPoint) -> None:
        """Update the optimizer with the result of a trial."""
        pass

    # TODO: Trial
    @abstractmethod
    def ask(self, n: int = 1) -> Union[SearchPoint, List[SearchPoint]]:
        """Ask for one or more candidate configurations to evaluate."""
        pass

    @classmethod
    def create(
        cls,
        space: SearchSpace,
        metrics: MetricSet,
        seed: Optional[int] = None,
    ) -> 'Optimizer':
        """Factory method to create an instance of the optimizer."""
        return cls(metrics=metrics, space=space, seed=seed)
