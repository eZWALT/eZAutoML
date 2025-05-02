from dataclasses import dataclass, field
from typing import Callable, Optional, Dict
from enum import Enum


# ===----------------------------------------------------------------------===#
# Metric & MetricSet                                                          #
#                                                                             #
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

class Comparison(str, Enum):
    BETTER = "better"
    WORSE = "worse"
    EQUAL = "equal"


@dataclass(frozen=True)
class Metric:
    """Represents an objective metric with direction and optional bounds."""
    name: str
    fn: Optional[Callable[..., float]] = field(default=None, compare=False)
    minimize: bool = True
    bounds: tuple[float, float] | None = None

    def evaluate(self, *args, **kwargs) -> float:
        """Evaluate the metric using the provided arguments."""
        if self.fn is None:
            raise ValueError(f"Metric '{self.name}' has no function attached.")
        return self.fn(*args, **kwargs)

    def is_improvement(self, current: float, challenger: float) -> Comparison:
        """Compares the current value with the challenger value."""
        if current == challenger:
            return Comparison.EQUAL
        if (challenger < current and self.minimize) or (challenger > current and not self.minimize):
            return Comparison.BETTER
        return Comparison.WORSE

    @property
    def optimal(self) -> float:
        """The optimal value of the metric (best value)."""
        if self.bounds:
            return self.bounds[0] if self.minimize else self.bounds[1]
        return float("-inf") if self.minimize else float("inf")

    @property
    def worst(self) -> float:
        """The worst possible value of the metric (worst value)."""
        if self.bounds:
            return self.bounds[1] if self.minimize else self.bounds[0]
        return float("inf") if self.minimize else float("-inf")


@dataclass(frozen=True)
class MetricSet:
    """A collection of multiple metrics, organized as a set."""
    metrics: Dict[str, Metric] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Metric:
        return self.metrics[key]

    def __iter__(self):
        return iter(self.metrics)

    def __len__(self):
        return len(self.metrics)
    
    def items(self):
        return self.metrics.items()

    def get_best_values(self) -> Dict[str, float]:
        return {k: v.optimal for k, v in self.metrics.items()}

    def get_worst_values(self) -> Dict[str, float]:
        return {k: v.worst for k, v in self.metrics.items()}


# Test features
if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
    import numpy as np


    metrics = {
        "accuracy": Metric(name="accuracy", fn=accuracy_score, minimize=False),
        "mse": Metric(name="mse", fn=mean_squared_error, minimize=True),
        "f1_score": Metric(name="f1_score", fn=f1_score, minimize=False)
    }

    # True and predicted values
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred_good = np.array([1, 0, 1, 1, 0])
    y_pred_bad = np.array([0, 0, 0, 0, 0])

    # Evaluate and compare metrics in a compact loop
    for metric_name, metric in metrics.items():
        score_good = metric.evaluate(y_true, y_pred_good)
        score_bad = metric.evaluate(y_true, y_pred_bad)
        print(f"{metric_name}: Good = {score_good}, Bad = {score_bad}, Improvement = {metric.is_improvement(score_good, score_bad).value}")