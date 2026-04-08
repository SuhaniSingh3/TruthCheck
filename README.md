# Fake News Detection System

A machine learning model for detecting fake news using scikit-learn, TF-IDF vectorization, and Logistic Regression. Includes a Flask REST API for predictions.

## Project Structure

```
fake-news-detection/
├── data/                          # Data directory
│   └── WELFake_dataset.csv       # Training dataset
├── models/                        # Trained models
│   ├── fake_news_model.pkl       # Serialized model
│   └── metrics.txt               # Model performance metrics
├── train.py                      # Training script
├── data_loader.py                # Data loading utilities
├── app.py                        # Flask API server
├── test_api.py                   # API testing script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Clone/Navigate to the project:**
```bash
cd fake-news-detection
```

2. **Create a virtual environment (optional but recommended):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

```bash
python train.py
```

This will:
- Load or generate the training dataset
- Create a TF-IDF vectorizer and Logistic Regression classifier
- Split data into 80% training, 20% testing
- Train the model and display performance metrics
- Save the model to `models/fake_news_model.pkl`
- Save metrics to `models/metrics.txt`

**Output Example:**
```
==============================================================
FAKE NEWS DETECTION MODEL - TRAINING
==============================================================

[1/5] Loading data...
Dataset Statistics:
Total samples: 100
Fake news (label=1): 50
Real news (label=0): 50

[2/5] Splitting data (80% train, 20% test)...
Training samples: 80
Testing samples: 20

[3/5] Creating ML pipeline...
Pipeline components:
  - TF-IDF Vectorizer (max_features=5000, ngram_range=(1,2))
  - Logistic Regression Classifier

[4/5] Training model...
✓ Model training completed!

[5/5] Evaluating model...

==============================================================
MODEL PERFORMANCE METRICS
==============================================================
Accuracy:  0.9500 (95.00%)
Precision: 0.9565
Recall:    0.9565
F1-Score:  0.9565

Confusion Matrix:
  True Negatives:  10
  False Positives: 0
  False Negatives: 1
  True Positives:  9
```

### Step 2: Start the Flask API

```bash
python app.py
```

The API will start at `http://localhost:5000`

**Available Endpoints:**

#### GET /
Home endpoint with service information
```bash
curl http://localhost:5000
```

#### GET /health
Health check
```bash
curl http://localhost:5000/health
```

#### GET /info
Model information and metrics
```bash
curl http://localhost:5000/info
```

#### POST /predict
Predict if a text is fake or real news
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Your news text here"}'
```

**Response Example:**
```json
{
  "success": true,
  "text": "New study shows benefits of exercise",
  "prediction": 0,
  "label": "REAL NEWS",
  "confidence": 92.45,
  "probabilities": {
    "real_news": 92.45,
    "fake_news": 7.55
  },
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

#### POST /batch_predict
Predict multiple texts at once
```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts":["text1", "text2", "text3"]}'
```

### Step 3: Test the API

While the Flask app is running, in another terminal:

```bash
python test_api.py
```

This runs a comprehensive test suite including:
- Health checks
- Model information retrieval
- Single predictions
- Batch predictions
- Error handling

## Model Details

### Algorithm
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
  - Max features: 5000
  - N-gram range: (1, 2) - unigrams and bigrams
  - Min document frequency: 2
  - Max document frequency: 0.8

- **Classifier**: Logistic Regression
  - Max iterations: 1000
  - Regularization parameter C: 1.0
  - Random state: 42

### Features
- Text preprocessing with TF-IDF
- Stop words removal (English)
- Lowercase normalization
- Unigram and bigram features

### Performance Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: Ratio of true positive fake news predictions
- **Recall**: Ratio of actual fake news detected
- **F1-Score**: Harmonic mean of precision and recall

### Dataset
The model trains on a dataset with:
- **Fake news samples**: News articles with false or misleading claims
- **Real news samples**: Factual news from reputable sources
- **Total samples**: ~100 (expandable with larger datasets)
- **Train/Test split**: 80/20

## Extending the Project

### Use Your Own Dataset

Modify `data_loader.py` to load your CSV file:
```python
def load_data():
    df = pd.read_csv('your_dataset.csv')
    # Ensure it has 'text' and 'label' columns (1=fake, 0=real)
    return df
```

### Improve Model Performance

1. **Larger Dataset**: Train on more samples
2. **Better Features**: Use Word2Vec, GloVe embeddings
3. **Deep Learning**: Try LSTM or BERT models
4. **Hyperparameter Tuning**: Use GridSearchCV for optimization
5. **Ensemble Methods**: Combine multiple models

### Add Database Integration

Store predictions and feedback:
```python
# In app.py
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String)
    label = db.Column(db.Integer)
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime)
```

### Deploy to Production

1. Use Gunicorn instead of Flask development server
2. Add authentication and rate limiting
3. Use environment variables for configuration
4. Deploy on AWS, GCP, or Azure

## Troubleshooting

### Model not found error
**Solution**: Run `python train.py` first to train and save the model

### Port 5000 already in use
**Solution**: Change the port in `app.py`
```python
app.run(host='localhost', port=5001)  # Use different port
```

### Memory issues with large datasets
**Solution**: Use `chunksize` parameter in pandas
```python
df = pd.read_csv('large_dataset.csv', chunksize=5000)
```

## Future Improvements

- [ ] Add more sophisticated NLP preprocessing
- [ ] Implement cross-validation for better evaluation
- [ ] Add LIME explanations for predictions
- [ ] Web UI dashboard for predictions
- [ ] Multi-language support
- [ ] Real-time model updates with feedback

## License

This project is open source and available under the MIT License.

## Author

Fake News Detection System v1.0 - Created with scikit-learn, Flask, and Python
