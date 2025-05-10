from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.text import Text
import json
import csv
from dataclasses import asdict, is_dataclass


from ezautoml.results.trial import Trial
from ezautoml.evaluation.evaluator import Evaluation

def to_json_serializable(obj):
    """Convert a dataclass or object to a JSON-serializable format."""
    if is_dataclass(obj):
        return {k: to_json_serializable(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    else:
        return obj

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
    
    def to_csv(self, filepath: str):
        """Save the trial history to a CSV file."""
        if not self.trials:
            return

        fieldnames = ["seed", "model", "optimizer", "duration"] + list(self.trials[0].evaluation.results.keys())

        with open(filepath, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trial in self.trials:
                row = {
                    "seed": trial.seed,
                    "model": trial.model_name,
                    "optimizer": trial.optimizer_name,
                    "duration": trial.duration
                }
                row.update(trial.evaluation.results)
                writer.writerow(row)
                
    def to_json(self, filepath: str):
        """Save the entire history (trials, evaluations) to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump([to_json_serializable(trial) for trial in self.trials], f, indent=4)

    @classmethod
    def from_json(cls, filepath: str) -> "History":
        """Load history from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        history = cls()
        for trial_data in data:
            # Deserialize the trial and evaluation
            evaluation = Evaluation(
                metric_set=MetricSet({}),  # Assuming MetricSet is initialized properly
                results=trial_data['evaluation']
            )
            trial = Trial(
                seed=trial_data['seed'],
                model_name=trial_data['model_name'],
                optimizer_name=trial_data['optimizer_name'],
                evaluation=evaluation,
                duration=trial_data['duration']
            )
            history.add(trial)

        return history
        
    


if __name__ == "__main__":
    from ezautoml.evaluation.evaluator import Evaluation
    from ezautoml.evaluation.metric import Metric, MetricSet
    from ezautoml.results.trial import Trial
    from ezautoml.results.history import History

    # Dummy function to create trials
    def make_trial(seed, acc):
        eval = Evaluation(metric_set=MetricSet({}), results={"accuracy": acc})
        return Trial(seed=seed, model_name=f"Model_{seed}", optimizer_name="Optuna", evaluation=eval, duration=0.01 * seed)

    # Create History and add trials
    history = History()
    for i in range(10):
        history.add(make_trial(seed=i, acc=0.8 + i * 0.01))

    # Display summary of the top 5 trials based on accuracy
    history.summary(metrics=["accuracy"])

    # Save the history to JSON and CSV files
    history.to_json("history.json")
    history.to_csv("history_summary.csv")
