import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_data():
    """Load iris dataset"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target, columns=['target'])
    return X, y, iris.target_names

def train_model():
    """Train machine learning model"""
    print("Loading data...")
    X, y, target_names = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values.ravel(), test_size=0.2, random_state=42
    )
    
    print("Training model...")
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/iris_model.pkl')
    joblib.dump(target_names, 'models/target_names.pkl')
    
    print("Model saved successfully!")
    return model, accuracy

if __name__ == "__main__":
    train_model()