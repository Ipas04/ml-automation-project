import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
from datetime import datetime

def save_metrics(accuracy, model, X_test, y_test, y_pred, target_names, model_name="iris_model"):
    """Save comprehensive training metrics to JSON"""
    try:
        # Calculate additional metrics
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(
            ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            model.feature_importances_.tolist()
        ))
        
        metrics = {
            'model_info': {
                'name': model_name,
                'algorithm': 'RandomForestClassifier',
                'n_estimators': model.n_estimators,
                'random_state': model.random_state
            },
            'dataset_info': {
                'name': 'iris',
                'n_samples': len(X_test) + len(model.classes_) * 40,  # Approximate total
                'n_features': X_test.shape[1],
                'n_classes': len(target_names),
                'classes': target_names.tolist(),
                'test_size': len(X_test)
            },
            'performance': {
                'accuracy': float(accuracy),
                'accuracy_percentage': float(accuracy * 100),
                'confusion_matrix': cm.tolist(),
                'feature_importance': feature_importance
            },
            'training_info': {
                'timestamp': datetime.now().isoformat(),
                'sklearn_version': __import__('sklearn').__version__,
                'python_version': __import__('sys').version.split()[0],
                'status': 'success'
            }
        }
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Save metrics
        metrics_path = 'models/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Metrics saved to {metrics_path}")
        
        # Verify and display key metrics
        print(f"📊 Key Metrics:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Classes: {', '.join(target_names)}")
        print(f"   Test samples: {len(X_test)}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error saving metrics: {e}")
        # Create minimal fallback metrics
        fallback_metrics = {
            'model_info': {'name': model_name, 'status': 'metrics_error'},
            'performance': {'accuracy': float(accuracy)},
            'training_info': {
                'timestamp': datetime.now().isoformat(),
                'status': 'success_with_metrics_error',
                'error': str(e)
            }
        }
        
        try:
            with open('models/metrics.json', 'w') as f:
                json.dump(fallback_metrics, f, indent=2)
            print("📄 Fallback metrics saved")
        except:
            print("❌ Could not save fallback metrics")
        
        return fallback_metrics

def load_data():
    """Load and prepare iris dataset"""
    try:
        print("📊 Loading iris dataset...")
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        
        print(f"✅ Dataset loaded:")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {list(X.columns)}")
        print(f"   Classes: {list(iris.target_names)}")
        
        return X, y, iris.target_names
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def train_model():
    """Train machine learning model with comprehensive JSON metrics"""
    try:
        print("🚀 Starting ML model training...")
        print("=" * 50)
        
        # Load data
        X, y, target_names = load_data()
        
        # Split data
        print("🔄 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
        
        # Train model
        print("🤖 Training RandomForest model...")
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10
        )
        model.fit(X_train, y_train)
        print("✅ Model training completed")
        
        # Make predictions
        print("🔮 Making predictions...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Display results
        print(f"\n📊 Model Performance:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Save model files
        print("💾 Saving model files...")
        os.makedirs('models', exist_ok=True)
        
        model_files = {
            'models/iris_model.pkl': model,
            'models/target_names.pkl': target_names
        }
        
        for filepath, data in model_files.items():
            joblib.dump(data, filepath)
            size = os.path.getsize(filepath)
            print(f"✅ {filepath} saved ({size} bytes)")
        
        # Save comprehensive metrics
        print("📄 Saving metrics to JSON...")
        metrics = save_metrics(accuracy, model, X_test, y_test, y_pred, target_names)
        
        # Verify all files
        print("\n🔍 Verifying saved files:")
        required_files = ['models/iris_model.pkl', 'models/target_names.pkl', 'models/metrics.json']
        
        all_good = True
        for filepath in required_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"✅ {filepath} ({size} bytes)")
            else:
                print(f"❌ {filepath} MISSING!")
                all_good = False
        
        if all_good:
            print("\n🎉 Model training completed successfully!")
            print("✅ All files saved and verified")
        else:
            print("\n⚠️  Some files are missing!")
            
        return model, accuracy, metrics
        
    except Exception as e:
        print(f"\n💥 Training failed with error: {e}")
        
        # Save error metrics
        try:
            os.makedirs('models', exist_ok=True)
            error_metrics = {
                'model_info': {'name': 'iris_model', 'status': 'failed'},
                'performance': {'accuracy': 0.0},
                'training_info': {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'failed',
                    'error': str(e),
                    'error_type': type(e).__name__
                }
            }
            
            with open('models/metrics.json', 'w') as f:
                json.dump(error_metrics, f, indent=2)
            print("📄 Error metrics saved")
            
        except Exception as save_error:
            print(f"❌ Could not save error metrics: {save_error}")
        
        raise

if __name__ == "__main__":
    try:
        model, accuracy, metrics = train_model()
        print(f"\n🏆 Final Result: {accuracy:.4f} accuracy")
    except Exception as e:
        print(f"\n💥 Script failed: {e}")
        exit(1)