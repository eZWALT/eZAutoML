from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ezautoml.results.trial import Trial

class History:
    def __init__(self):
        self.trials: List[Trial] = []

    def add(self, trial: Trial):
        """Add a trial to the history."""
        self.trials.append(trial)

    def best(self, metric: str = "accuracy") -> Optional[Trial]:
        """Return the trial with the best performance on the given metric."""
        valid_trials = [t for t in self.trials if t.evaluation and metric in t.evaluation.results]
        return max(valid_trials, key=lambda t: t.evaluation.results.get(metric, float('-inf')), default=None)

    def top_k(self, k: int = 5, metric: str = "accuracy") -> List[Trial]:
        """Return the top k trials based on the given metric."""
        valid_trials = [t for t in self.trials if metric in t.evaluation.results]
        return sorted(valid_trials, key=lambda t: t.evaluation.results.get(metric, float('-inf')), reverse=True)[:k]


    def summary(self, k: int = 10, metrics: List[str] = ["accuracy", "f1_score"]):
        """Pretty print the top k trials with rich library, enhanced version with multiple metrics."""
        console = Console()
        table = Table(title=f"Top {k} Trials", show_lines=True)

        # Add columns dynamically based on metrics
        table.add_column("Rank", justify="right")
        table.add_column("Seed")
        table.add_column("Model")
        table.add_column("Optimizer")
        table.add_column("Duration (s)")
        for metric in metrics:
            table.add_column(metric.capitalize(), justify="center")

        for i, trial in enumerate(self.top_k(k), start=1):
            row = [str(i), str(trial.seed), trial.model_name, trial.optimizer_name, f"{trial.duration:.2f}"]

            # Add scores for each metric
            for metric in metrics:
                score = trial.evaluation.results.get(metric, "N/A")
                if isinstance(score, float):
                    score = f"{score:.4f}"
                row.append(score)

            # Highlight the best trial with color (e.g., green for highest accuracy)
            if i == 1:  # Highlight top trial with green color
                row = [Text(item, style="bold green") for item in row]
            table.add_row(*row)

        console.print(table)


if __name__ == "__main__":
    from types import SimpleNamespace
    from ezautoml.evaluation.evaluation import Evaluation
    from ezautoml.evaluation.metric import Metric, MetricSet
    from sklearn.metrics import accuracy_score

    # Dummy function to create trials
    def make_trial(seed, acc):
        eval = Evaluation(MetricSet({}), results={"accuracy": acc})
        return Trial(seed=seed, model_name=f"Model_{seed}", optimizer_name="Optuna", evaluation=eval, duration=0.01 * seed)

    # Create History
    history = History()
    for i in range(10):
        history.add(make_trial(seed=i, acc=0.8 + i * 0.01))

    # Display summary of the top 5 trials based on accuracy
    history.summary(metrics=["accuracy"])
