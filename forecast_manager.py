import pandas as pd
from typing import List, Dict, Optional

from data_collector import DataCollector
from bayesian_model import BayesianModel
from question import Question
from data_point import DataPoint

class ForecastManager:
    def __init__(self):
        self.collector = DataCollector()
        self.model = BayesianModel()
        self.questions: Dict[str, Question] = {}
    
    def add_question(self, question: Question):
        
        data = self.collector.collect_data(question)
        question.historical_data = data
        self.questions[question.id] = question
        
        self.model.train(data)
        prediction = self.model.predict(data)
        question.current_prediction = prediction
        
        print(f"\nInitial prediction for '{question.text}':")
        print(f"Probability: {prediction:.2%}")
        
        # Print the data we used
        historical_df = pd.DataFrame([
            {
                'Date': d.timestamp,
                'Event': d.source,
                'Value': d.value,
                'Confidence': d.confidence
            } for d in data
        ])
        print("\nHistorical and Expert Data Used:")
        print(historical_df)
    
    def update_prediction(self, question_id: str, new_evidence: List[DataPoint]):
        question = self.questions[question_id]
        question.historical_data.extend(new_evidence)
        
        self.model.train(question.historical_data)
        prediction = self.model.predict(new_evidence)
        question.current_prediction = prediction
        
        print(f"\nUpdated prediction for '{question.text}':")
        print(f"New probability: {prediction:.2%}")