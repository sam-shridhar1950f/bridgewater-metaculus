from datetime import datetime
from dataclasses import dataclass

@dataclass
class DataPoint:
    timestamp: datetime
    value: float  # probability between 0-1
    source: str
    confidence: float  # confidence intervals (0-1)
