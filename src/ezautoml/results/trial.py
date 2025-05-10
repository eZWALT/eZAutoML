from dataclasses import dataclass, asdict
from typing import Dict, Any
import time
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from ezautoml.evaluation.evaluator import Evaluation


# TODO add also feature_processors, data_processors,
# feature_engineering, opt_algorithm_selection
@dataclass
class Trial:
    seed: int
    model_name: str
    optimizer_name: str
    evaluation: Evaluation  # <-- Now storing an Evaluation object
    duration: float  # in seconds

    def print_summary(self) -> None:
        """Pretty print the trial using rich."""
        table = Table.grid(padding=(0, 1))
        table.add_row("Seed", str(self.seed))
        table.add_row("Model", self.model_name)
        table.add_row("Optimizer", self.optimizer_name)
        table.add_row("Evaluation", str(self.evaluation))  # Uses __str__ from Evaluation
        table.add_row("Duration", f"{self.duration:.2f} seconds")

        panel = Panel(table, title=f"Trial Summary (Seed: {self.seed})", title_align="left")
        Console().print(panel)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the trial to a dictionary."""
        d = asdict(self)
        d["evaluation"] = self.evaluation.results  # Flatten for dict export
        return d

    def __str__(self) -> str:
        return f"[Trial seed={self.seed}, model={self.model_name}, optimizer={self.optimizer_name}, {self.evaluation}, duration={self.duration:.2f}s]"

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    from ezautoml.evaluation.metric import MetricSet

    # Create a dummy evaluation object
    dummy_results = {"accuracy": 0.912, "f1_score": 0.880}
    dummy_metric_set = MetricSet(metrics={}, primary_metric_name="accuracy")  
    evaluation = Evaluation(results=dummy_results, metric_set=dummy_metric_set)

    # Create and display the Trial
    trial = Trial(
        seed=42,
        model_name="ResNet50",
        optimizer_name="Adam",
        evaluation=evaluation,
        duration=420.3
    )

    print(str(trial))      # Pretty summary
    trial.print_summary()  # Rich terminal panel
