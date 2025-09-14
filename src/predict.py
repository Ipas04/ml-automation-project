import joblib
import numpy as np
import pandas as pd

class IrisPredictor:
    def __init__(self, model_path='models/iris_model.pkl', target_names_path='models/target_names.pkl'):
        self.model = joblib.load(model_path)
        self.target_names = joblib.load(target_names_path)
    
    def predict(self, features):
        """
        Predict iris species
        features: list or array of [sepal_length, sepal_width, petal_length, petal_width]
        """
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'prediction': self.target_names[prediction],
            'probabilities': {
                name: float(prob) for name, prob in zip(self.target_names, probability)
            }
        }

def main():
    # Example prediction
    predictor = IrisPredictor()
    
    # Sample data: [sepal_length, sepal_width, petal_length, petal_width]
    sample_features = [5.1, 3.5, 1.4, 0.2]
    result = predictor.predict(sample_features)
    
    print(f"Prediction: {result['prediction']}")
    print("Probabilities:")
    for species, prob in result['probabilities'].items():
        print(f"  {species}: {prob:.4f}")

if __name__ == "__main__":
    main()