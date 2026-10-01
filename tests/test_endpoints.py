import unittest
import json
from unittest.mock import patch

from app import create_app
import extensions


class EndpointTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_health(self):
        res = self.client.get('/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('groq_active', data)

    def test_predict_missing_text(self):
        res = self.client.post('/predict', json={})
        self.assertEqual(res.status_code, 400)

    def test_predict_short_text(self):
        res = self.client.post('/predict', json={'text': 'sh'})
        self.assertEqual(res.status_code, 400)

    def test_smart_analyze_endpoint(self):
        res = self.client.post('/api/analyze', json={'text': 'Scientists reveal new telescope image capturing deep universe galaxies.'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('success'))
        self.assertIn('label', data)
        self.assertIn('confidence', data)

    def test_analyze_route_alias(self):
        res = self.client.post('/analyze', json={'headline': 'Government announces nationwide road safety initiative.'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('success'))

    @patch('services.groq_service.predict_news')
    def test_predict_mocked_success(self, mock_predict):
        mock_predict.return_value = {
            'label': 'FAKE NEWS',
            'prediction': 1,
            'confidence': 95.0,
            'reasons': ['mocked reason'],
            'summary': 'mocked summary'
        }
        res = self.client.post('/predict', json={'text': 'This is a sufficiently long test text for prediction.'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('success'))
        self.assertEqual(data.get('label'), 'FAKE NEWS')


    def test_clear_history(self):
        res = self.client.post('/api/clear-history')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('success'))
        self.assertIn('History cleared', data.get('message', ''))


if __name__ == '__main__':
    unittest.main()
