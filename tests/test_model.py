import pytest
import numpy as np
import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import load_data, train_model
from predict import IrisPredictor

def test_load_data():
    """Test data loading"""
    X, y, target_names = load_data()
    
    assert X.shape[0] == 150  # Iris dataset has 150 samples
    assert X.shape[1] == 4    # 4 features
    assert len(target_names) == 3  # 3 classes
    assert len(y) == 150

def test_model_training():
    """Test model training with JSON metrics"""
    # Updated to handle 3 return values
    model, accuracy, metrics = train_model()
    
    # Test return values
    assert accuracy > 0.8  # Model should have decent accuracy
    assert model is not None  # Model should exist
    assert metrics is not None  # Metrics should exist
    
    # Test files exist
    assert os.path.exists('models/iris_model.pkl')
    assert os.path.exists('models/target_names.pkl')
    assert os.path.exists('models/metrics.json')
    
    # Test JSON metrics content
    with open('models/metrics.json', 'r') as f:
        saved_metrics = json.load(f)
    
    # Validate JSON structure
    assert 'model_info' in saved_metrics
    assert 'performance' in saved_metrics
    assert 'training_info' in saved_metrics
    
    # Validate performance metrics
    assert saved_metrics['performance']['accuracy'] == accuracy
    assert saved_metrics['training_info']['status'] == 'success'
    
    print(f"✅ Model training test passed with accuracy: {accuracy:.4f}")

def test_prediction():
    """Test model prediction"""
    # Ensure model exists first
    if not os.path.exists('models/iris_model.pkl'):
        train_model()
    
    predictor = IrisPredictor()
    
    # Test with sample data
    sample_features = [5.1, 3.5, 1.4, 0.2]
    result = predictor.predict(sample_features)
    
    assert 'prediction' in result
    assert 'probabilities' in result
    assert 'confidence' in result  # New field
    assert len(result['probabilities']) == 3
    assert sum(result['probabilities'].values()) == pytest.approx(1.0, rel=1e-5)
    assert 0.0 <= result['confidence'] <= 1.0
    
    print(f"✅ Prediction test passed: {result['prediction']} with {result['confidence']:.4f} confidence")

def test_json_metrics_validation():
    """Test JSON metrics file validation"""
    # Ensure model is trained
    if not os.path.exists('models/metrics.json'):
        train_model()
    
    # Load and validate JSON structure
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Test required fields
    required_sections = ['model_info', 'performance', 'training_info', 'dataset_info']
    for section in required_sections:
        assert section in metrics, f"Missing section: {section}"
    
    # Test model_info
    model_info = metrics['model_info']
    assert 'name' in model_info
    assert 'algorithm' in model_info
    assert model_info['algorithm'] == 'RandomForestClassifier'
    
    # Test performance
    performance = metrics['performance']
    assert 'accuracy' in performance
    assert 'feature_importance' in performance
    assert 0.0 <= performance['accuracy'] <= 1.0
    
    # Test training_info
    training_info = metrics['training_info']
    assert 'timestamp' in training_info
    assert 'status' in training_info
    assert training_info['status'] == 'success'
    
    # Test dataset_info
    dataset_info = metrics['dataset_info']
    assert dataset_info['name'] == 'iris'
    assert dataset_info['n_classes'] == 3
    assert dataset_info['n_features'] == 4
    
    print("✅ JSON metrics validation passed")

def test_model_performance_threshold():
    """Test model meets performance threshold"""
    # Load metrics
    if not os.path.exists('models/metrics.json'):
        train_model()
    
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    accuracy = metrics['performance']['accuracy']
    min_threshold = 0.8
    
    assert accuracy >= min_threshold, f"Accuracy {accuracy:.4f} below threshold {min_threshold}"
    
    print(f"✅ Performance test passed: {accuracy:.4f} >= {min_threshold}")

if __name__ == "__main__":
    # Run tests individually for debugging
    print("🧪 Running individual tests...")
    
    try:
        print("\n1. Testing data loading...")
        test_load_data()
        
        print("\n2. Testing model training...")
        test_model_training()
        
        print("\n3. Testing predictions...")
        test_prediction()
        
        print("\n4. Testing JSON metrics...")
        test_json_metrics_validation()
        
        print("\n5. Testing performance threshold...")
        test_model_performance_threshold()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()