from typing import Dict, Any 
from dataclasses import dataclass

import time 
import uuid

from eZAutoML.results.outcome import Outcome

@dataclass 
class Trial: 
    id: str 
    config: Dict[str, Any]
    outcome: Outcome 
    timestamp: float
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], outcome: Outcome) -> "Trial":
        return cls(
            id=str(uuid.uuid4()),
            config=config,
            outcome=outcome,
            timestamp=time.time(),
        )
    