import os
import sys
import pytest
import requests
from time import sleep

# Add project root to path
sys.path.insert(0, os.getcwd())

# Test configuration
AUTH_URL = "http://auth_service:8001"
PREDICT_URL = "http://predict_service:8002"

# Test credentials
TEST_USER = "admin"
TEST_PASSWORD = "admin123"

@pytest.fixture(scope="session")
def auth_token():
    """Get authentication token for test session"""
    # Wait for services to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            # Try auth service health check
            resp = requests.get(f"{AUTH_URL}/health", timeout=5)
            if resp.status_code == 200:
                break
        except:
            pass
        print(f"Waiting for auth service... ({i+1}/{max_retries})")
        sleep(2)
    
    # Wait for predict service
    for i in range(max_retries):
        try:
            resp = requests.get(f"{PREDICT_URL}/health", timeout=5)
            if resp.status_code == 200:
                break
        except:
            pass
        print(f"Waiting for predict service... ({i+1}/{max_retries})")
        sleep(2)
    
    # Login to get token
    login_data = {
        "username": TEST_USER,
        "password": TEST_PASSWORD
    }
    resp = requests.post(
        f"{AUTH_URL}/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    token = resp.json()["access_token"]
    return token

def test_auth_service_health():
    """Test auth service health endpoint"""
    resp = requests.get(f"{AUTH_URL}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_predict_service_health():
    """Test predict service health endpoint"""
    resp = requests.get(f"{PREDICT_URL}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_login_success():
    """Test successful login"""
    login_data = {
        "username": TEST_USER,
        "password": TEST_PASSWORD
    }
    resp = requests.post(
        f"{AUTH_URL}/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert resp.status_code == 200
    assert "access_token" in resp.json()
    assert resp.json()["token_type"] == "bearer"

def test_login_failure():
    """Test failed login with wrong credentials"""
    login_data = {
        "username": "wrong_user",
        "password": "wrong_pass"
    }
    resp = requests.post(
        f"{AUTH_URL}/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert resp.status_code == 401

def test_predict_without_auth():
    """Test prediction without authentication"""
    test_data = {
        "designation": "Test Product",
        "description": "This is a test description"
    }
    resp = requests.post(f"{PREDICT_URL}/predict", json=test_data)
    assert resp.status_code == 422  

def test_predict_with_invalid_token():
    """Test prediction with invalid token"""
    test_data = {
        "designation": "Test Product",
        "description": "This is a test description"
    }
    headers = {"token": "invalid_token_here"}
    resp = requests.post(f"{PREDICT_URL}/predict", json=test_data, headers=headers)
    assert resp.status_code == 401

def test_predict_success(auth_token):
    """Test successful prediction"""
    test_data = {
        "designation": "Test Product",
        "description": "This is a test description for a product"
    }
    headers = {"token": auth_token}
    resp = requests.post(f"{PREDICT_URL}/predict", json=test_data, headers=headers)
    assert resp.status_code == 200
    result = resp.json()
    assert "predicted_class" in result
    assert isinstance(result["predicted_class"], str)
    print(f"Prediction result: {result}")

def test_predict_empty_description(auth_token):
    """Test prediction with empty description"""
    test_data = {
        "designation": "Test Product",
        "description": ""
    }
    headers = {"token": auth_token}
    resp = requests.post(f"{PREDICT_URL}/predict", json=test_data, headers=headers)
    assert resp.status_code == 200
    assert "predicted_class" in resp.json()

def test_predict_special_characters(auth_token):
    """Test prediction with special characters"""
    test_data = {
        "designation": "Test!@# Product$%^",
        "description": "Description with special chars: €£¥ & symbols"
    }
    headers = {"token": auth_token}
    resp = requests.post(f"{PREDICT_URL}/predict", json=test_data, headers=headers)
    assert resp.status_code == 200
    assert "predicted_class" in resp.json()

def test_model_info(auth_token):
    """Test model info endpoint"""
    headers = {"token": auth_token}
    resp = requests.get(f"{PREDICT_URL}/model-info", headers=headers)
    assert resp.status_code == 200
    info = resp.json()
    assert "model_version" in info
    assert "parameters" in info
    assert "metrics" in info
    print(f"Model info: {info}")

