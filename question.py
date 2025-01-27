from data_point import DataPoint
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

@dataclass
class Question:
    id: str
    text: str
    category: str  
    deadline: datetime
    historical_data: List[DataPoint] = None
    current_prediction: float = None