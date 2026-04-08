"""
Quick Reference Guide for Fake News Detection System
"""

QUICK_COMMANDS = """
╔════════════════════════════════════════════════════════════════╗
║  FAKE NEWS DETECTION SYSTEM - QUICK REFERENCE                  ║
╚════════════════════════════════════════════════════════════════╝

1. INITIAL SETUP
   ────────────────────────────────────────────────────────────
   pip install -r requirements.txt
   
   Or use the quick start script (Windows):
   quick_start.bat


2. TRAIN THE MODEL
   ────────────────────────────────────────────────────────────
   python train.py
   
   Creates: models/fake_news_model.pkl
   Also: models/metrics.txt (performance metrics)


3. START THE API SERVER
   ────────────────────────────────────────────────────────────
   python app.py
   
   Server starts at: http://localhost:5000


4. TEST THE API (run in another terminal while server is running)
   ────────────────────────────────────────────────────────────
   python test_api.py
   
   Runs complete test suite


5. MAKE PREDICTIONS (without API - direct model usage)
   ────────────────────────────────────────────────────────────
   python example_predictions.py
   
   Uses model directly for batch predictions


╔════════════════════════════════════════════════════════════════╗
║  API ENDPOINTS                                                  ║
╚════════════════════════════════════════════════════════════════╝

GET  /                 - Service info and available endpoints

GET  /health           - Health check
     Response: {"status": "healthy", "model_loaded": true}

GET  /info             - Model metrics and information
     Response: Model performance metrics

POST /predict          - Single text prediction
     Request:  {"text": "Your news text here"}
     Response: {
       "label": "REAL NEWS" or "FAKE NEWS",
       "confidence": 92.45,
       "probabilities": {"real_news": 92.45, "fake_news": 7.55}
     }

POST /batch_predict    - Predict multiple texts
     Request:  {"texts": ["text1", "text2", "text3"]}
     Response: List of predictions for each text


╔════════════════════════════════════════════════════════════════╗
║  CURL EXAMPLES                                                  ║
╚════════════════════════════════════════════════════════════════╝

# Single Prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"New economic policy announced by government"}'

# Batch Prediction
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts":["text1", "text2", "text3"]}'

# Health Check
curl http://localhost:5000/health

# Model Info
curl http://localhost:5000/info


╔════════════════════════════════════════════════════════════════╗
║  PROJECT STRUCTURE                                              ║
╚════════════════════════════════════════════════════════════════╝

fake-news-detection/
├── train.py              - Training script
├── app.py                - Flask API server
├── test_api.py           - API test suite
├── example_predictions.py - Example predictions (no API)
├── data_loader.py        - Data loading utilities
├── config.py             - Configuration and constants
├── requirements.txt      - Python dependencies
├── quick_start.bat       - Quick start script (Windows)
│
├── data/                 - Data directory
│   └── WELFake_dataset.csv (generated)
│
├── models/               - Trained models
│   ├── fake_news_model.pkl (generated)
│   └── metrics.txt (generated)
│
└── README.md             - Full documentation


╔════════════════════════════════════════════════════════════════╗
║  MODEL ARCHITECTURE                                             ║
╚════════════════════════════════════════════════════════════════╝

Pipeline:
  Input Text
    ↓
  TF-IDF Vectorizer
    • Max features: 5000
    • N-grams: (1,2) - unigrams + bigrams
    • Stop words removed
    ↓
  Logistic Regression Classifier
    • C parameter: 1.0
    • Max iterations: 1000
    ↓
  Output: [0=Real, 1=Fake] + Confidence


╔════════════════════════════════════════════════════════════════╗
║  PERFORMANCE METRICS                                            ║
╚════════════════════════════════════════════════════════════════╝

After training, metrics are saved to models/metrics.txt

Key metrics:
  • Accuracy  - % of correct predictions
  • Precision - % of predicted fakes that are actually fake
  • Recall    - % of actual fakes that were detected
  • F1-Score  - Harmonic mean of Precision and Recall


╔════════════════════════════════════════════════════════════════╗
║  TROUBLESHOOTING                                                ║
╚════════════════════════════════════════════════════════════════╝

❌ "Model not found" error
   → Run: python train.py

❌ "Port 5000 already in use"
   → Edit: app.py - change port in: app.run(..., port=5001)

❌ Import errors after pip install
   → Try: pip install --upgrade setuptools wheel

❌ Connection refused when testing API
   → Make sure Flask server is running in another terminal
   → Check that http://localhost:5000 is accessible


╔════════════════════════════════════════════════════════════════╗
║  EXTENDING THE PROJECT                                          ║
╚════════════════════════════════════════════════════════════════╝

1. Use Your Own Dataset
   - Prepare CSV file with 'text' and 'label' columns
   - Modify data_loader.py to load your file

2. Improve Model Performance
   - Increase dataset size
   - Try different algorithms (SVM, Random Forest)
   - Use embeddings (Word2Vec, GloVe, BERT)
   - Tune hyperparameters with GridSearchCV

3. Add Web Interface
   - Create HTML/CSS/JS frontend
   - Integrate with Flask

4. Database Integration
   - Store predictions and feedback
   - Track model performance over time

5. Production Deployment
   - Use Gunicorn instead of Flask dev server
   - Add authentication and rate limiting
   - Deploy on AWS, GCP, or Azure


╔════════════════════════════════════════════════════════════════╗
║  PYTHON VERSION & DEPENDENCIES                                  ║
╚════════════════════════════════════════════════════════════════╝

Required: Python 3.7+

Key Libraries:
  • scikit-learn >= 1.3.0   (ML algorithms)
  • pandas >= 2.0.3         (Data processing)
  • numpy >= 1.24.3         (Numerical computing)
  • flask >= 2.3.2          (Web framework)
  • nltk >= 3.8.1           (NLP utilities)
  • joblib >= 1.3.1         (Model serialization)


════════════════════════════════════════════════════════════════
For more information, see README.md
════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(QUICK_COMMANDS)
