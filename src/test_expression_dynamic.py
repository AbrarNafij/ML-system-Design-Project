import random
import requests

# Endpoints
BASE_URL = "http://127.0.0.1:8001"
SAMPLES_URL = f"{BASE_URL}/samples"
EXPRESSION_URL = f"{BASE_URL}/predict/expression"

print("🔍 Step 1: Fetching valid sample IDs from the database...")
try:
    samples_resp = requests.get(SAMPLES_URL)
    if samples_resp.status_code != 200:
        print(f"❌ Failed to get samples. Status code: {samples_resp.status_code}")
        exit()
        
    samples_data = samples_resp.json()
    # Grab the first available real sample ID (e.g., "GSM119615")
    real_sample_id = samples_data["samples"][0]["sample_id"]
    print(f"✅ Found real sample ID: {real_sample_id}")

    # Note: Since your API handles sample extraction in predictor.py, 
    # we can simulate the 5,000 exact probes by using a mock dictionary populated 
    # with the explicit error examples and matching keys if available, 
    # or testing the sample endpoint directly.
    
    # If your pipeline requires testing the raw expression dictionary directly,
    # we need the exact feature index array. Let's see if we can trigger a prediction 
    # by mocking a broader dictionary or utilizing the sample pipeline.
    
    print(f"\n📡 Testing /predict/sample directly to verify the pipeline...")
    sample_payload = {"sample_id": real_sample_id}
    sample_resp = requests.post(f"{BASE_URL}/predict/sample", json=sample_payload)
    print(f"[Sample Predict Response]: {sample_resp.status_code}")
    print(sample_resp.json())

except requests.exceptions.ConnectionError:
    print(f"❌ Connection Error: Is your FastAPI server running on {BASE_URL}?")