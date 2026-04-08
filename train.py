"""
Training script for fake news detection model using scikit-learn
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
from data_loader import load_data

def train_model():
    """Train the fake news detection model"""
    
    print("=" * 60)
    print("FAKE NEWS DETECTION MODEL - TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    
    # Split data
    print("\n[2/5] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create pipeline
    print("\n[3/5] Creating ML pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        ))
    ])
    
    print("Pipeline components:")
    print("  - TF-IDF Vectorizer (max_features=5000, ngram_range=(1,2))")
    print("  - Logistic Regression Classifier")
    
    # Train model
    print("\n[4/5] Training model...")
    pipeline.fit(X_train, y_train)
    print("[OK] Model training completed!")
    
    # Make predictions
    print("\n[5/5] Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    # Save model
    print("\n" + "=" * 60)
    os.makedirs('models', exist_ok=True)
    model_path = 'models/fake_news_model.pkl'
    joblib.dump(pipeline, model_path)
    print(f"[OK] Model saved to: {model_path}")
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    metrics_path = 'models/metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("FAKE NEWS DETECTION MODEL - METRICS\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"[OK] Metrics saved to: {metrics_path}")
    print("=" * 60)
    
    return pipeline, metrics

if __name__ == "__main__":
    train_model()
