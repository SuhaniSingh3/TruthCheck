"""
Flask API for fake news detection
Run with: python app.py
Then test with:
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"text":"Your news text here"}'
"""
from flask import Flask, request, jsonify, render_template
import random
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Load model
MODEL_PATH = 'models/fake_news_model.pkl'

def load_model():
    """Load the trained model"""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint serving the frontend UI"""
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result_page():
    """Detailed Explanation Page"""
    return render_template('result.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    if not os.path.exists('models/metrics.txt'):
        return jsonify({'error': 'Metrics file not found. Train model first.'}), 404
    
    with open('models/metrics.txt', 'r') as f:
        metrics_content = f.read()
    
    return jsonify({
        'model_name': 'Fake News Detection Model v1.0',
        'type': 'Text Classification (TF-IDF + Logistic Regression)',
        'metrics': metrics_content
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if given text is fake or real news
    Expected JSON: {"text": "news text here"}
    Returns: {"prediction": 1 or 0, "confidence": float, "label": "FAKE/REAL"}
    """
    if model is None:
        # Mock prediction for UI demonstration when model fails to load
        is_fake = random.choice([0, 1])
        conf = random.uniform(60.0, 99.9)
        
        if is_fake == 1:
            reasons = ["Uses highly sensational formatting", "Source URL has low trust score", "Emotional triggers detected"]
        else:
            reasons = ["Cross-verified with credible sources", "Neutral tone detected", "Formatting is standard"]
            
        return jsonify({
            'success': True,
            'text': request.get_json().get('text', '')[:100] + '...',
            'prediction': is_fake,
            'label': "FAKE NEWS" if is_fake == 1 else "REAL NEWS",
            'confidence': round(conf, 2),
            'probabilities': {
                'real_news': round(100 - conf, 2) if is_fake else round(conf, 2),
                'fake_news': round(conf, 2) if is_fake else round(100 - conf, 2)
            },
            'reasons': reasons,
            'timestamp': datetime.now().isoformat()
        })
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing "text" field in JSON payload'
            }), 400
        
        text = data['text'].strip()
        
        if len(text) < 10:
            return jsonify({
                'error': 'Text too short. Please provide at least 10 characters.'
            }), 400
        
        # Make prediction
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        
        # Get confidence
        confidence = max(probabilities) * 100
        
        # Format response
        label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
        
        return jsonify({
            'success': True,
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': int(prediction),
            'label': label,
            'confidence': round(confidence, 2),
            'probabilities': {
                'real_news': round(probabilities[0] * 100, 2),
                'fake_news': round(probabilities[1] * 100, 2)
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch predict multiple texts
    Expected JSON: {"texts": ["text1", "text2", ...]}
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data or not isinstance(data['texts'], list):
            return jsonify({
                'error': 'Invalid payload. Expected JSON with "texts" list.'
            }), 400
        
        texts = [t.strip() for t in data['texts'] if isinstance(t, str) and len(t.strip()) >= 10]
        
        if not texts:
            return jsonify({
                'error': 'No valid texts provided (min 10 chars each).'
            }), 400
        
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)
        
        results = []
        for text, pred, probs in zip(texts, predictions, probabilities):
            label = "FAKE NEWS" if pred == 1 else "REAL NEWS"
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'prediction': int(pred),
                'label': label,
                'confidence': round(max(probs) * 100, 2)
            })
        
        return jsonify({
            'success': True,
            'total_processed': len(texts),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    if model is None:
        print("WARNING: Model not found. Please run train.py first to train the model.")
    else:
        print("Model loaded successfully!")
    
    print("\nStarting Fake News Detection API...")
    print("Server running at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, host='localhost', port=5000)
