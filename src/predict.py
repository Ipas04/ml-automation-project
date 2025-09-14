import joblib
import numpy as np
import pandas as pd
import json
import os

class IrisPredictor:
    def __init__(self, model_path='models/iris_model.pkl', target_names_path='models/target_names.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.target_names = joblib.load(target_names_path)
            
            # Load metrics if available
            self.metrics = self.load_metrics()
            
            print("✅ Model loaded successfully")
            if self.metrics:
                accuracy = self.metrics.get('performance', {}).get('accuracy', 'N/A')
                print(f"📊 Model accuracy: {accuracy}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_metrics(self):
        """Load training metrics from JSON"""
        try:
            metrics_path = 'models/metrics.json'
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                print("📄 Metrics loaded from JSON")
                return metrics
            else:
                print("⚠️  No metrics file found")
                return None
        except Exception as e:
            print(f"⚠️  Could not load metrics: {e}")
            return None
    
    def get_model_info(self):
        """Get model information from metrics"""
        if not self.metrics:
            return "No metrics available"
        
        info = []
        if 'model_info' in self.metrics:
            model_info = self.metrics['model_info']
            info.append(f"Model: {model_info.get('name', 'Unknown')}")
            info.append(f"Algorithm: {model_info.get('algorithm', 'Unknown')}")
        
        if 'performance' in self.metrics:
            perf = self.metrics['performance']
            accuracy = perf.get('accuracy', 0)
            info.append(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if 'training_info' in self.metrics:
            training = self.metrics['training_info']
            info.append(f"Trained: {training.get('timestamp', 'Unknown')}")
        
        return "\n".join(info)
    
    def predict(self, features):
        """
        Predict iris species
        features: list or array of [sepal_length, sepal_width, petal_length, petal_width]
        """
        try:
            features = np.array(features).reshape(1, -1)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            result = {
                'prediction': self.target_names[prediction],
                'confidence': float(max(probabilities)),
                'probabilities': {
                    name: float(prob) for name, prob in zip(self.target_names, probabilities)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise

def main():
    """Test predictions with multiple samples"""
    try:
        print("🔮 Iris Species Predictor")
        print("=" * 40)
        
        # Initialize predictor
        predictor = IrisPredictor()
        
        # Display model info
        print(f"\n📋 Model Information:")
        print(predictor.get_model_info())
        
        # Test samples
        test_samples = [
            {
                'name': 'Small flower',
                'features': [4.5, 2.8, 1.3, 0.3],
                'expected': 'setosa'
            },
            {
                'name': 'Medium flower', 
                'features': [6.0, 3.0, 4.5, 1.5],
                'expected': 'versicolor'
            },
            {
                'name': 'Large flower',
                'features': [7.2, 3.2, 6.0, 2.0],
                'expected': 'virginica'
            }
        ]
        
        print(f"\n🧪 Testing Predictions:")
        print("-" * 40)
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\nTest {i}: {sample['name']}")
            print(f"Features: {sample['features']}")
            
            result = predictor.predict(sample['features'])
            
            print(f"🔮 Prediction: {result['prediction']}")
            print(f"🎯 Confidence: {result['confidence']:.4f}")
            print(f"📊 All probabilities:")
            
            for species, prob in result['probabilities'].items():
                indicator = "👑" if species == result['prediction'] else "  "
                print(f"   {indicator} {species}: {prob:.4f}")
            
            # Check if prediction matches expected
            if result['prediction'].lower() == sample['expected'].lower():
                print("✅ Correct prediction!")
            else:
                print(f"❌ Expected: {sample['expected']}")
        
        print(f"\n🎉 Prediction testing completed!")
        
    except Exception as e:
        print(f"💥 Prediction test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)