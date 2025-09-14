import os
import json
from datetime import datetime

def save_metrics(accuracy, model_name="iris_model"):
    """Save training metrics"""
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'timestamp': datetime.now().isoformat(),
        'dataset': 'iris'
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def load_metrics():
    """Load training metrics"""
    try:
        with open('models/metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None