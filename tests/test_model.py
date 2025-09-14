import pytest
import numpy as np
import os
import sys

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
    """Test model training"""
    model, accuracy = train_model()
    
    assert accuracy > 0.8  # Model should have decent accuracy
    assert os.path.exists('models/iris_model.pkl')
    assert os.path.exists('models/target_names.pkl')

def test_prediction():
    """Test model prediction"""
    # First ensure model exists
    if not os.path.exists('models/iris_model.pkl'):
        train_model()
    
    predictor = IrisPredictor()
    
    # Test with sample data
    sample_features = [5.1, 3.5, 1.4, 0.2]
    result = predictor.predict(sample_features)
    
    assert 'prediction' in result
    assert 'probabilities' in result
    assert len(result['probabilities']) == 3
    assert sum(result['probabilities'].values()) == pytest.approx(1.0, rel=1e-5)