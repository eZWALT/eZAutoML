from typing import Dict
from dataclasses import dataclass, field
from ezautoml.evaluation.metric import Metric, MetricSet

# ===----------------------------------------------------------------------===#
# Evaluation                                                                  #
#                                                                             #
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

@dataclass
class Evaluation:
    """Class responsible for evaluating predictions using a MetricSet."""
    metric_set: MetricSet
    results: Dict[str, float] = field(default_factory=dict)

    def evaluate(self, ground_truth: 'ArrayLike', predictions: 'ArrayLike') -> Dict[str, float]:
        """Evaluate predictions using the metrics in the MetricSet."""
        self.results = {
            metric_name: metric.evaluate(ground_truth, predictions)
            for metric_name, metric in self.metric_set.items()
        }
        return self.results

    def compare(self, other_evaluation: 'Evaluation') -> Dict[str, str]:
        """Compare this evaluation with another evaluation."""
        comparison = {}
        for metric_name in self.results:
            current_value = self.results[metric_name]
            challenger_value = other_evaluation.results.get(metric_name)

            if challenger_value is None:
                comparison[metric_name] = "missing in challenger"
                continue

            # Compare the current result with the challenger result
            improvement = self.metric_set[metric_name].is_improvement(current_value, challenger_value)
            comparison[metric_name] = improvement.value

        return comparison

    def get_results(self) -> Dict[str, float]:
        """Return the current evaluation results."""
        return self.results

    def __str__(self) -> str:
        if not self.results:
            return "No results"
        return ", ".join(f"{k}: {v:.4f}" for k, v in self.results.items())



if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
    # Define a set of metrics
    metrics = MetricSet(metrics={
        "accuracy": Metric(name="accuracy", fn=accuracy_score, minimize=False),
        "mse": Metric(name="mse", fn=mean_squared_error, minimize=True),
        "f1_score": Metric(name="f1_score", fn=f1_score, minimize=False)
    })

    # Create an evaluator instance
    evaluator = Evaluation(metric_set=metrics)

    # True and predicted values
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred_good = np.array([1, 0, 1, 1, 0])  # Good predictions
    y_pred_bad = np.array([0, 0, 0, 0, 0])   # Bad predictions

    # Evaluate and compare metrics in a compact loop
    evaluator.evaluate(y_true, y_pred_good)
    good_eval = evaluator.get_results()
    print(f"Good predictions: {good_eval}")

    evaluator.evaluate(y_true, y_pred_bad)
    bad_eval = evaluator.get_results()
    print(f"Bad predictions: {bad_eval}")

    # Compare the evaluations
    comparison = evaluator.compare(Evaluation(metric_set=metrics, results=good_eval))
    print(f"Comparison: {comparison}")