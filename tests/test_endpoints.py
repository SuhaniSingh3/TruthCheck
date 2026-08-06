import json
import pytest

from app import create_app


def test_health():
    app = create_app()
    client = app.test_client()
    res = client.get('/health')
    assert res.status_code == 200
    data = res.get_json()
    assert data['status'] == 'healthy'
    assert 'groq_active' in data


def test_predict_missing_text():
    app = create_app()
    client = app.test_client()
    res = client.post('/predict', json={})
    assert res.status_code == 400


def test_predict_short_text():
    app = create_app()
    client = app.test_client()
    res = client.post('/predict', json={'text': 'short'})
    assert res.status_code == 400


def test_predict_success(monkeypatch):
    app = create_app()
    client = app.test_client()

    # Mock the external Groq call to return deterministic output
    fake_result = {
        'label': 'FAKE NEWS',
        'prediction': 1,
        'confidence': 0.9,
        'reasons': ['mocked reason'],
        'summary': 'mocked summary'
    }

    monkeypatch.setattr('app.predict_with_groq', lambda text: fake_result)

    # Prevent DB commits from interfering with tests
    import extensions
    monkeypatch.setattr(extensions.db.session, 'add', lambda obj: None)
    monkeypatch.setattr(extensions.db.session, 'commit', lambda: None)

    res = client.post('/predict', json={'text': 'This is a sufficiently long test text for prediction.'})
    assert res.status_code == 200
    data = res.get_json()
    assert data.get('success') is True
    assert data.get('label') == 'FAKE NEWS'
