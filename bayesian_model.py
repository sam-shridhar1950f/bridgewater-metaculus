from data_point import DataPoint
import numpy as np
from typing import List, Dict, Optional

class BayesianModel:
    def __init__(self):
        self.prior = None
        self.likelihood = None
    
    def train(self, data: List[DataPoint]):
        # Calculate prior from historical events
        values = [d.value for d in data]
        confidences = [d.confidence for d in data]
        self.prior = np.average(values, weights=confidences)
        self.likelihood = np.std(values)  # Use standard deviation as likelihood
    
    def predict(self, new_evidence: List[DataPoint]) -> float:
        if not self.prior:
            return None
            
        # If no new evidence, return prior
        if not new_evidence:
            return self.prior
            
        # Apply Bayes theorem with evidence
        evidence_values = [d.value for d in new_evidence]
        evidence_conf = [d.confidence for d in new_evidence]
        likelihood = np.average(evidence_values, weights=evidence_conf)
        
        posterior = (likelihood * self.prior) / np.mean(evidence_values)
        return min(max(posterior, 0), 1)