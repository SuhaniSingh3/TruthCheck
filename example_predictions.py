"""
Example prediction script - demonstrates direct model usage without API
"""
import joblib
import os
from config import PATHS, LABELS

def predict_from_model(text):
    """
    Load model directly and make prediction
    Useful for batch processing or non-API usage
    """
    # Check if model exists
    if not os.path.exists(PATHS['model_file']):
        print(f"Error: Model not found at {PATHS['model_file']}")
        print("Please run 'python train.py' first to train the model.")
        return None
    
    # Load model
    print("Loading model...")
    model = joblib.load(PATHS['model_file'])
    
    # Make prediction
    print(f"\nAnalyzing text: '{text[:80]}...'")
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    
    # Format result
    label = LABELS[int(prediction)]
    confidence = max(probabilities) * 100
    
    result = {
        'text': text,
        'prediction': int(prediction),
        'label': label,
        'confidence': confidence,
        'real_news_probability': probabilities[0] * 100,
        'fake_news_probability': probabilities[1] * 100,
    }
    
    return result

def main():
    """Run example predictions"""
    
    test_texts = [
        "Scientists announce breakthrough in renewable energy technology",
        "President secretly controls weather using satellites says anonymous source",
        "New study links diet to improved mental health outcomes",
        "Ancient astronauts built the Great Wall of China according to leaked document",
        "Stock market reaches new highs on strong earnings reports",
    ]
    
    print("="*70)
    print("FAKE NEWS DETECTION - DIRECT MODEL PREDICTION EXAMPLES")
    print("="*70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Predicting: '{text}'")
        print("-" * 70)
        
        result = predict_from_model(text)
        
        if result:
            print(f"   Label: {result['label']}")
            print(f"   Confidence: {result['confidence']:.2f}%")
            print(f"   Real News Probability: {result['real_news_probability']:.2f}%")
            print(f"   Fake News Probability: {result['fake_news_probability']:.2f}%")

if __name__ == "__main__":
    main()
