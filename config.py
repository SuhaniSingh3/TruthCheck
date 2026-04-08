"""
Configuration and constants for the Fake News Detection system
"""

# Model Configuration
MODEL_CONFIG = {
    'max_features': 5000,      # Maximum number of features from TF-IDF
    'min_df': 2,               # Minimum document frequency
    'max_df': 0.8,             # Maximum document frequency ratio
    'ngram_range': (1, 2),     # Unigrams and bigrams
    'stop_words': 'english',   # Stop words to filter
}

# Classifier Configuration
CLASSIFIER_CONFIG = {
    'max_iter': 1000,
    'random_state': 42,
    'C': 1.0,
}

# API Configuration
API_CONFIG = {
    'host': 'localhost',
    'port': 5000,
    'debug': True,
}

# Data Configuration
DATA_CONFIG = {
    'test_size': 0.2,
    'train_size': 0.8,
    'random_state': 42,
    'stratify': True,
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'min_text_length': 10,  # Minimum characters for prediction
    'confidence_threshold': 0.5,
}

# Label Mapping
LABELS = {
    0: 'REAL NEWS',
    1: 'FAKE NEWS',
}

# Paths
PATHS = {
    'data': 'data',
    'models': 'models',
    'model_file': 'models/fake_news_model.pkl',
    'metrics_file': 'models/metrics.txt',
}
