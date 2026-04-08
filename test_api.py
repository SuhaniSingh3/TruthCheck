"""
Test script for the Fake News Detection API
Run this after starting the Flask app (python app.py)
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_info():
    """Test info endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Model Information")
    print("="*60)
    response = requests.get(f"{BASE_URL}/info")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_single_prediction():
    """Test single prediction"""
    print("\n" + "="*60)
    print("TEST 3: Single Prediction")
    print("="*60)
    
    test_texts = [
        "Scientists discover new renewable energy technology that could revolutionize power generation",
        "Secret government experiment proves that birds are actually robots spying on citizens"
    ]
    
    for text in test_texts:
        print(f"\nText: {text[:80]}...")
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Prediction: {result.get('label')}")
        print(f"Confidence: {result.get('confidence')}%")

def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*60)
    print("TEST 4: Batch Prediction")
    print("="*60)
    
    texts = [
        "Federal Reserve announces new economic policy measures",
        "Aliens have secretly taken over world governments according to leaked documents",
        "Study shows positive correlation between exercise and mental health",
        "New 5G towers will control your thoughts through microwave radiation"
    ]
    
    response = requests.post(
        f"{BASE_URL}/batch_predict",
        json={"texts": texts}
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total Processed: {result.get('total_processed')}")
    print("\nResults:")
    for i, res in enumerate(result.get('results', []), 1):
        print(f"  {i}. {res['label']} (Confidence: {res['confidence']}%)")

def test_error_handling():
    """Test error handling"""
    print("\n" + "="*60)
    print("TEST 5: Error Handling")
    print("="*60)
    
    # Missing text field
    print("\n5a. Missing 'text' field:")
    response = requests.post(f"{BASE_URL}/predict", json={})
    print(f"Status: {response.status_code}")
    print(f"Error: {response.json().get('error')}")
    
    # Text too short
    print("\n5b. Text too short:")
    response = requests.post(f"{BASE_URL}/predict", json={"text": "short"})
    print(f"Status: {response.status_code}")
    print(f"Error: {response.json().get('error')}")

if __name__ == "__main__":
    print("FAKE NEWS DETECTION API - TEST SUITE")
    print("Make sure the Flask app is running (python app.py)\n")
    
    try:
        test_health()
        test_info()
        test_single_prediction()
        test_batch_prediction()
        test_error_handling()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60 + "\n")
    
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the Flask app is running:")
        print("  python app.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
