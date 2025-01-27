import os
from anthropic import Anthropic, APIError
from datetime import datetime
import pandas as pd

def test_llm_collection():
    try:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not found")
        llm = Anthropic(api_key=api_key)
    except (ValueError, APIError) as e:
        print(f"Error initializing Anthropic client: {str(e)}")
        raise
        
    question = "Will Google, Meta, Amazon, Tesla, or X accept crypto as payment by December 31, 2025?"
    
    historical_prompt = f"""You are a forecasting expert. For the question: {question}
    List 10 similar historical events or precedents that could inform our prediction.
    
    For each event, provide:
    1. Date in YYYY-MM-DD format
    2. Brief description of the event
    3. Outcome (1 if similar outcome occurred, 0 if it didn't)
    4. Your confidence in the relevance (0-1)
    
    Format exactly as:
    YYYY-MM-DD | EVENT | OUTCOME | CONFIDENCE
    
    Provide only the formatted lines, no additional text."""

    try:
        historical_response = llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            messages=[{"role": "user", "content": historical_prompt}]
        )
        
        # Print raw response for debugging
        print("Raw Historical Response:")
        print(historical_response.content[0].text)
        print("\n" + "="*50 + "\n")
        
        historical_lines = [line.strip() for line in historical_response.content[0].text.split('\n') if '|' in line]
        historical_data = []
        
        for line in historical_lines:
            try:
                date, event, outcome, confidence = [x.strip() for x in line.split('|')]
                historical_data.append({
                    'Date': pd.to_datetime(date),
                    'Event': event,
                    'Outcome': float(outcome),
                    'Confidence': float(confidence)
                })
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(f"Error: {str(e)}")
                continue
                
        historical_df = pd.DataFrame(historical_data)
        print("Historical Data DataFrame:")
        print(historical_df)
        
    except Exception as e:
        print(f"Error processing historical data: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    expert_prompt = f"""You are a forecasting expert. For the question: {question}
    Based on your April 2024 knowledge, provide 5 expert predictions.
    
    For each prediction include:
    1. Expert name/source
    2. Probability prediction (0-1)
    3. Confidence in expertise (0-1)
    4. Brief reasoning
    
    Format exactly as:
    EXPERT | PREDICTION | CONFIDENCE | REASONING
    
    Provide only the formatted lines, no additional text."""

    try:
        expert_response = llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            messages=[{"role": "user", "content": expert_prompt}]
        )
        
        # Print raw response for debugging
        print("Raw Expert Response:")
        print(expert_response.content[0].text)
        print("\n" + "="*50 + "\n")
        
        expert_lines = [line.strip() for line in expert_response.content[0].text.split('\n') if '|' in line]
        expert_data = []
        
        for line in expert_lines:
            try:
                expert, prediction, confidence, reasoning = [x.strip() for x in line.split('|')]
                expert_data.append({
                    'Expert': expert,
                    'Prediction': float(prediction),
                    'Confidence': float(confidence),
                    'Reasoning': reasoning
                })
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(f"Error: {str(e)}")
                continue
                
        expert_df = pd.DataFrame(expert_data)
        print("Expert Opinion DataFrame:")
        print(expert_df)
        
    except Exception as e:
        print(f"Error processing expert data: {str(e)}")

if __name__ == "__main__":
    test_llm_collection()