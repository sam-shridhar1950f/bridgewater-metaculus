from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum
import os
from anthropic import Anthropic, APIError

from data_point import DataPoint
from data_collector import DataCollector
from bayesian_model import BayesianModel
from question import Question
from forecast_manager import ForecastManager

def main():
    manager = ForecastManager()
    
    q = Question(
        id="crypto-1",
        text="Will Google, Meta, Amazon, Tesla, or X accept crypto as payment by December 31, 2025?",
        category="tech",
        deadline=datetime(2025, 12, 31)
    )
    manager.add_question(q)
    
    # # Example of adding new evidence later
    # new_data = [
    #     DataPoint(
    #         timestamp=datetime.now(),
    #         value=0.8,
    #         source="Google files cryptocurrency payment patent",
    #         confidence=0.9
    #     )
    # ]
    new_data = []
    manager.update_prediction("crypto-1", new_data)

if __name__ == "__main__":
    main()