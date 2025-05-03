from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import uuid
from enum import Enum
from rich.panel import Panel
from rich.table import Table
import time

from ezautoml.evaluation.evaluation import Evaluation

# ===----------------------------------------------------------------------===#
# Optimization Trial                                                          #
#                                                                             #
# This class describes all information related to an attempt of optimization  #
# performed by an optimizer. How long it lasted, its evaluation... and can be #
# pretty printed into terminal by using rich library                          #
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

class TrialStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Trial:
    def __init__(self, 
                 seed: int, 
                 optimizer: str, 
                 evaluation: Dict[str, float], 
                 duration: float):
        # Essential attributes
        self.seed = seed
        self.optimizer = optimizer
        self.evaluation = evaluation
        self.duration = duration  # Duration is passed externally (end_time - start_time)

    @classmethod
    def create(cls, seed: int, optimizer: str) -> "Trial":
        return cls(seed=seed, optimizer=optimizer, evaluation={}, duration=0.0)

    def print_summary(self, start_time: float, end_time: float, evaluation_results: Dict[str, float]) -> None:
        """Pretty print the trial details using the rich library."""
        table = Table.grid(padding=(0, 1))
        table.add_row("Seed", str(self.seed))
        table.add_row("Optimizer", self.optimizer)
        table.add_row("Start Time", time.ctime(start_time))
        table.add_row("End Time", time.ctime(end_time))
        table.add_row("Evaluation", str(evaluation_results))
        table.add_row("Duration", f"{self.duration:.4f} seconds")

        panel = Panel(table, title=f"Trial Summary ({self.seed})", title_align="left")
        
        # Print the panel to console
        from rich.console import Console
        console = Console()
        console.print(panel)

    def to_dict(self, start_time: float, end_time: float, evaluation_results: Dict[str, float]) -> Dict[str, Any]:
        """Return the trial as a dictionary (useful for saving the trial details)."""
        return {
            "seed": self.seed,
            "optimizer": self.optimizer,
            "evaluation": evaluation_results,
            "start_time": start_time,
            "end_time": end_time,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trial":
        """Create a trial from a dictionary (useful for loading saved trials)."""
        return cls(
            seed=data["seed"],
            optimizer=data["optimizer"],
            evaluation=data["evaluation"],
            duration=data["duration"]
        )