import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing /health endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("✓ Health check passed")

def test_info():
    """Test info endpoint"""
    print("\n=== Testing /info endpoint ===")
    response = requests.get(f"{BASE_URL}/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    assert "team" in response.json()
    assert response.json()["team"] == "macbuntu"
    print("✓ Info endpoint passed")

def test_predict_valid():
    """Test predict endpoint with valid data"""
    print("\n=== Testing /predict with valid data ===")
    payload = {
        "features": {"pclass": 3, "sex": "male", "age": 59.0, "sibsp": 0, "parch": 0, "fare": 7.25, "embarked": "S"}
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert "prediction" in response.json()
    print("✓ Valid prediction passed")

def test_predict_missing_fields():
    """Test predict endpoint with missing fields"""
    print("\n=== Testing /predict with missing fields ===")
    payload = {
        "features": {
            "Pclass": 1,
            "Sex": "female"
        }
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("✓ Missing fields test completed")

def test_predict_invalid_data():
    """Test predict endpoint with invalid data"""
    print("\n=== Testing /predict with invalid data ===")
    payload = {
        "features": {
            "Pclass": "invalid",
            "Sex": 123,
            "Age": "not_a_number"
        }
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("✓ Invalid data test completed")

def test_train():
    """Test train endpoint"""
    print("\n=== Testing /train endpoint ===")
    response = requests.get(f"{BASE_URL}/train")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✓ Train endpoint passed")

def test_predict_female_first_class():
    """Test prediction for female first class passenger"""
    print("\n=== Testing /predict - Female, 1st class ===")
    payload = {
        "features": {
            "Pclass": 1,
            "Sex": "female",
            "Age": 35,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 80,
            "Embarked": "C"
        }
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Response: {response.json()}")
    print("✓ Female first class prediction completed")

def test_predict_outlier_age():
    """Test prediction with outlier age"""
    print("\n=== Testing /predict with outlier age ===")
    payload = {
        "features": {
            "Pclass": 2,
            "Sex": "male",
            "Age": 150,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 15,
            "Embarked": "Q"
        }
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Response: {response.json()}")
    print("✓ Outlier age test completed")

if __name__ == "__main__":
    print("Starting API tests...")
    print(f"Base URL: {BASE_URL}")
    
    try:
        test_health()
        test_info()
        test_predict_valid()
        test_predict_female_first_class()
        test_predict_missing_fields()
        test_predict_outlier_age()
        test_predict_invalid_data()
        test_train()
        
        print("\n" + "="*50)
        print("All tests completed!")
        print("="*50)
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API. Make sure the server is running.")
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")