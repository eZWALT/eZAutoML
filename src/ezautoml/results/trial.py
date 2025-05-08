from dataclasses import dataclass, asdict
from typing import Dict, Any
import time
from rich.panel import Panel
from rich.table import Table
from rich.console import Console


# TODO add also feature_processors, data_processors,
# feature_engineering, opt_algorithm_selection
@dataclass
class Trial:
    seed: int
    model_name: str
    optimizer_name: str
    evaluation: Dict[str, float]
    duration: float  # in seconds

    def print_summary(self) -> None:
        """Pretty print the trial using rich."""
        table = Table.grid(padding=(0, 1))
        table.add_row("Seed", str(self.seed))
        table.add_row("Model", self.model_name)
        table.add_row("Optimizer", self.optimizer_name)
        table.add_row("Evaluation", ", ".join(f"{k}={v:.3f}" for k, v in self.evaluation.items()))
        table.add_row("Duration", f"{self.duration:.2f} seconds")

        panel = Panel(table, title=f"Trial Summary (Seed: {self.seed})", title_align="left")
        Console().print(panel)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the trial to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trial":
        """Create a Trial instance from a dictionary."""
        return cls(**data)

    def __str__(self) -> str:
        """Compact summary of the trial."""
        metrics = ", ".join(f"{k}={v:.3f}" for k, v in self.evaluation.items())
        return f"[Trial seed={self.seed}, model={self.model_name}, optimizer={self.optimizer_name}, {metrics}, duration={self.duration:.2f}s]"

    def __repr__(self) -> str:
        """Diagnostic representation."""
        return self.__str__()

if __name__ == "__main__":
    trial = Trial(
        seed=42,
        model_name="ResNet50",
        optimizer_name="Adam",
        evaluation={"accuracy": 0.912, "f1_score": 0.880},
        duration=420.3
    )

    print(str(trial))      # Pretty summary
    trial.print_summary()  # Rich terminal panel
