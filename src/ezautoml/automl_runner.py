from ezautoml.optimization.optimizer import Optimizer
from ezautoml.space.search_point import SearchPoint
from ezautoml.results.trial import Trial
from ezautoml.evaluation.evaluator import Evaluator
from loguru import logger
from typing import List


class AutoMLRunner:
    def __init__(self, optimizer: Optimizer, evaluator: Evaluator) -> None:
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.history: List[SearchPoint] = []

    def run(self, n_trials: int) -> None:
        for i in range(n_trials):
            logger.info(f"[RUNNER] Trial {i+1}/{n_trials}")
            point = self.optimizer.ask()
            if not point:
                logger.warning("Stopping condition met.")
                break
            trial = self.evaluator.evaluate(point)
            self.optimizer.tell(trial)
            self.history.append(trial)

    def get_best(self) -> SearchPoint:
        return self.optimizer.get_best_trial()

    def get_history(self) -> List[SearchPoint]:
        return self.history
