from abc import ABC
from ezautoml.evaluation.task import TaskType

class Component:
    def __init__(self, name, constructor, hyperparams=None, task=TaskType.CLASSIFICATION):
        self.name = name
        self.constructor = constructor
        self.hyperparams = hyperparams or []
        self.task = task  # Now uses TaskType enum

    def sample_params(self):
        return {hp.name: hp.sample() for hp in self.hyperparams}

    def instantiate(self, params):
        return self.constructor(**params)

    def is_compatible(self, task: TaskType):
        return self.task == TaskType.BOTH or self.task == task


