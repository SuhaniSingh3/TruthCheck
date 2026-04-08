"""
Flask API for TruthCheck - Fake News Detection
Powered by Groq Cloud (Llama 3.3)
"""
from flask import Flask, request, jsonify, render_template
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Groq API Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

def predict_with_groq(text):
    """Predict news authenticity using Groq API (Llama-3.3)"""
    if not client:
        return None
    
    system_prompt = (
        "You are an expert news fact-checker. Analyze the provided news text and determine if it is REAL or FAKE.\n"
        "Respond ONLY in JSON format with these exact keys:\n"
        '{"label": "FAKE NEWS" or "REAL NEWS", "prediction": 1 or 0, "confidence": float, "reasons": [list of strings], "summary": "string"}\n'
        "Use prediction 1 for FAKE and 0 for REAL."
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this news: {text[:4000]}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Groq API Error: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result_page():
    return render_template('result.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'groq_active': client is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400
        
        text = data['text'].strip()
        if len(text) < 10:
            return jsonify({'error': 'Text too short'}), 400
        
        # Groq Prediction
        result = predict_with_groq(text)
        if result:
            return jsonify({
                'success': True,
                'source': 'Groq (Llama-3.3)',
                'text': text[:150] + '...' if len(text) > 150 else text,
                **result,
                'timestamp': datetime.now().isoformat()
            })

        return jsonify({'error': 'Prediction service unavailable. Please check GROQ_API_KEY.'}), 503
    
    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500

if __name__ == '__main__':
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not found in .env")
    
    print("\nTruthCheck - Starting API Server...")
    print("Server running at http://localhost:5000\n")
    app.run(debug=True, host='localhost', port=5000)
