import os
import json
from datetime import datetime
from typing import Dict, Optional

def save_metrics(accuracy: float, model_name: str = "iris_model", **kwargs) -> Dict:
    """Save training metrics to JSON with additional info"""
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'timestamp': datetime.now().isoformat(),
        'dataset': 'iris',
        **kwargs  # Additional metrics
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def load_metrics() -> Optional[Dict]:
    """Load training metrics from JSON"""
    try:
        with open('models/metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  No metrics file found")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing metrics JSON: {e}")
        return None

def display_metrics_summary(metrics: Dict) -> None:
    """Display formatted metrics summary"""
    if not metrics:
        print("No metrics available")
        return
    
    print("📊 Model Metrics Summary:")
    print("-" * 30)
    
    # Basic info
    if 'model_info' in metrics:
        model_info = metrics['model_info']
        print(f"Model: {model_info.get('name', 'Unknown')}")
        print(f"Algorithm: {model_info.get('algorithm', 'Unknown')}")
    
    # Performance
    if 'performance' in metrics:
        perf = metrics['performance']
        accuracy = perf.get('accuracy', 0)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Training info
    if 'training_info' in metrics:
        training = metrics['training_info']
        print(f"Status: {training.get('status', 'Unknown')}")
        print(f"Trained: {training.get('timestamp', 'Unknown')}")

def validate_model_performance(min_accuracy: float = 0.8) -> bool:
    """Validate model performance from JSON metrics"""
    metrics = load_metrics()
    if not metrics:
        print("❌ No metrics available for validation")
        return False
    
    accuracy = metrics.get('performance', {}).get('accuracy', 0)
    status = metrics.get('training_info', {}).get('status', 'unknown')
    
    print(f"🎯 Validating model performance...")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Status: {status}")
    print(f"   Threshold: {min_accuracy:.4f}")
    
    if status != 'success':
        print(f"❌ Training status: {status}")
        return False
    
    if accuracy < min_accuracy:
        print(f"❌ Accuracy {accuracy:.4f} below threshold {min_accuracy:.4f}")
        return False
    
    print(f"✅ Model validation passed!")
    return True

if __name__ == "__main__":
    # Test utilities
    metrics = load_metrics()
    if metrics:
        display_metrics_summary(metrics)
        validate_model_performance()
    else:
        print("No metrics found to display")