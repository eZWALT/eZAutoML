from typing import Dict
from dataclasses import dataclass, field
from ezautoml.evaluation.metric import Metric, MetricSet


@dataclass
class Evaluation:
    """Class to store evaluation results and perform comparisons."""
    results: Dict[str, float]
    metric_set: MetricSet

    def compare(self, other: 'Evaluation') -> Dict[str, str]:
        """Compare this evaluation with another evaluation."""
        comparison = {}
        for metric_name in self.results:
            current_value = self.results[metric_name]
            challenger_value = other.results.get(metric_name)

            if challenger_value is None:
                comparison[metric_name] = "missing in challenger"
                continue

            # Compare the current result with the challenger result
            improvement = self.metric_set[metric_name].is_improvement(current_value, challenger_value)
            comparison[metric_name] = improvement.value

        return comparison

    def __str__(self) -> str:
        if not self.results:
            return "No results"
        return ", ".join(f"{k}: {v:.4f}" for k, v in self.results.items())


class Evaluator:
    """Class responsible for evaluating predictions."""
    
    def __init__(self, metric_set: MetricSet):
        self.metric_set = metric_set
    
    def evaluate(self, ground_truth: 'ArrayLike', predictions: 'ArrayLike') -> Evaluation:
        """Evaluate predictions using the metrics in the MetricSet."""
        results = {
            metric_name: metric.evaluate(ground_truth, predictions)
            for metric_name, metric in self.metric_set.items()
        }
        return Evaluation(results, self.metric_set)


# Example usage:

if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
    
    # Define a set of metrics
    metrics = MetricSet(metrics={
        "accuracy": Metric(name="accuracy", fn=accuracy_score, minimize=False),
        "mse": Metric(name="mse", fn=mean_squared_error, minimize=True),
        "f1_score": Metric(name="f1_score", fn=f1_score, minimize=False)
    },
    primary_metric_name="accuracy")

    # Create an evaluator instance
    evaluator = Evaluator(metric_set=metrics)

    # True and predicted values
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred_good = np.array([1, 0, 1, 1, 0])  # Good predictions
    y_pred_bad = np.array([0, 0, 0, 0, 0])   # Bad predictions

    # Evaluate good predictions
    evaluation_good = evaluator.evaluate(y_true, y_pred_good)
    print(f"Good predictions: {evaluation_good}")

    # Evaluate bad predictions
    evaluation_bad = evaluator.evaluate(y_true, y_pred_bad)
    print(f"Bad predictions: {evaluation_bad}")

    # Compare the evaluations
    comparison = evaluation_good.compare(evaluation_bad)
    print(f"Comparison: {comparison}")
