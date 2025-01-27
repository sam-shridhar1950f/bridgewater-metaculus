from data_point import DataPoint
from anthropic import Anthropic, APIError
from question import Question
from typing import List, Dict, Optional
import os
import pandas as pd
from datetime import datetime

class DataCollector:
    def __init__(self):
        try:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not found")
            self.llm = Anthropic(api_key=api_key)
        except (ValueError, APIError) as e:
            print(f"Error initializing Anthropic client: {str(e)}")
            raise
    
    def collect_data(self, question: Question) -> List[DataPoint]:
        historical = self._get_historical_data(question)
        expert = self._get_expert_opinions(question)
        return historical + expert
    
    def _get_historical_data(self, question: Question) -> List[DataPoint]:
        prompt = f"""You are a forecasting expert. For the question: {question.text}
        List 10 similar historical events or precedents that could inform our prediction.
        
        For each event, provide:
        1. Date in YYYY-MM-DD format
        2. Brief description of the event
        3. Outcome (1 if similar outcome occurred, 0 if it didn't)
        4. Your confidence in the relevance (0-1)
        
        Format exactly as:
        YYYY-MM-DD | EVENT | OUTCOME | CONFIDENCE
        
        Provide only the formatted lines, no additional text."""
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        data_points = []
        lines = [line.strip() for line in response.content[0].text.split('\n') if '|' in line]
        
        for line in lines:
            try:
                date, event, outcome, confidence = [x.strip() for x in line.split('|')]
                data_points.append(DataPoint(
                    timestamp=pd.to_datetime(date),
                    value=float(outcome),
                    source=event,
                    confidence=float(confidence)
                ))
            except Exception as e:
                print(f"Error parsing historical line: {line}")
                print(f"Error: {str(e)}")
                continue
        
        return data_points
    
    def _get_expert_opinions(self, question: Question) -> List[DataPoint]:
        prompt = f"""You are a forecasting expert. For the question: {question.text}
        Based on your April 2024 knowledge, provide 5 expert predictions.
        
        For each prediction include:
        1. Expert name/source
        2. Probability prediction (0-1)
        3. Confidence in expertise (0-1)
        4. Brief reasoning
        
        Format exactly as:
        EXPERT | PREDICTION | CONFIDENCE | REASONING
        
        Provide only the formatted lines, no additional text."""
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        data_points = []
        lines = [line.strip() for line in response.content[0].text.split('\n') if '|' in line]
        
        for line in lines:
            try:
                expert, prediction, confidence, reasoning = [x.strip() for x in line.split('|')]
                data_points.append(DataPoint(
                    timestamp=datetime.now(),
                    value=float(prediction),
                    source=f"{expert}: {reasoning}",
                    confidence=float(confidence)
                ))
            except Exception as e:
                print(f"Error parsing expert line: {line}")
                print(f"Error: {str(e)}")
                continue
        
        return data_points