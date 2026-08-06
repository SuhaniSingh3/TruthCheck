"""
Report Model — TruthCheck Analysis History
Stores every analysis result for history, favorites, and report generation.
"""
from extensions import db
from datetime import datetime
import json


class Report(db.Model):
    """A single analysis report storing input, result, and metadata."""

    __tablename__ = 'reports'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)

    # --- Input Data ---
    input_type = db.Column(db.String(20), nullable=False)  # text, url, youtube, image, video
    input_text = db.Column(db.Text, nullable=True)          # Raw input text / URL
    input_title = db.Column(db.String(500), nullable=True)  # Article title if provided
    input_filename = db.Column(db.String(255), nullable=True)  # Uploaded file name

    # --- Analysis Result ---
    prediction = db.Column(db.String(50), nullable=True)     # FAKE NEWS, REAL NEWS, etc.
    confidence = db.Column(db.Float, nullable=True)           # 0-100 confidence score
    risk_level = db.Column(db.String(20), nullable=True)      # low, medium, high, critical
    reliability_score = db.Column(db.Float, nullable=True)    # 0-100 reliability score
    result_json = db.Column(db.Text, nullable=True)           # Full JSON result blob

    # --- Multilingual ---
    detected_language = db.Column(db.String(10), nullable=True)   # Auto-detected language code
    response_language = db.Column(db.String(10), default='en')    # User's preferred response lang

    # --- User Actions ---
    is_favorite = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text, nullable=True)                 # User notes on this report

    # --- Metadata ---
    analysis_duration = db.Column(db.Float, nullable=True)    # Seconds taken for analysis
    source = db.Column(db.String(50), nullable=True)          # e.g., "Groq (Llama-3.3)"
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # ---- Helpers ----

    def get_result(self):
        """Deserialize the full JSON result."""
        if self.result_json:
            try:
                return json.loads(self.result_json)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def set_result(self, result_dict):
        """Serialize a result dictionary to JSON for storage."""
        self.result_json = json.dumps(result_dict, ensure_ascii=False)

    @property
    def is_fake(self):
        """Quick check if prediction indicates fake content."""
        return self.prediction and 'FAKE' in self.prediction.upper()

    @property
    def verdict_class(self):
        """CSS class for styling the verdict."""
        if not self.prediction:
            return 'result-unknown'
        upper = self.prediction.upper()
        if 'FAKE' in upper:
            return 'result-fake'
        elif 'REAL' in upper:
            return 'result-real'
        elif 'MISLEADING' in upper or 'PARTIAL' in upper:
            return 'result-warning'
        return 'result-unknown'

    @property
    def risk_class(self):
        """CSS class for styling the risk level."""
        mapping = {
            'low': 'risk-low',
            'medium': 'risk-medium',
            'high': 'risk-high',
            'critical': 'risk-critical',
        }
        return mapping.get((self.risk_level or '').lower(), 'risk-unknown')

    def to_dict(self):
        """Convert report to a dictionary for JSON API responses."""
        return {
            'id': self.id,
            'input_type': self.input_type,
            'input_text': self.input_text,
            'input_title': self.input_title,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'risk_level': self.risk_level,
            'reliability_score': self.reliability_score,
            'result': self.get_result(),
            'detected_language': self.detected_language,
            'response_language': self.response_language,
            'is_favorite': self.is_favorite,
            'analysis_duration': self.analysis_duration,
            'source': self.source,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f'<Report {self.id} [{self.input_type}] {self.prediction}>'
