from enum import Enum

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    BOTH = "both"
