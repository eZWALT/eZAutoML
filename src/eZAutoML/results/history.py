from typing import List, Optional
from eZAutoML.results.trial import Trial


class History:
    def __init__(self):
        self.trials: List[Trial] = []

    def add(self, trial: Trial):
        self.trials.append(trial)

    def best(self, key: str = "score") -> Optional[Trial]:
        return max(self.trials, key=lambda t: getattr(t.outcome, key), default=None)

    def summary(self) -> str:
        return f"{len(self.trials)} trials, best score: {self.best().outcome.score if self.best() else 'N/A'}"
