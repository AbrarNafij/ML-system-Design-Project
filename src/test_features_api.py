import random
import requests

# 1. Define the API endpoint
URL = "http://127.0.0.1:8001/predict/features"

# 2. Generate exactly 5000 random float values
print("Generating 5,000 synthetic features...")
dummy_features = [round(random.uniform(-2.0, 2.0), 4) for _ in range(5000)]

# 3. Construct the JSON payload mapping to your schema
payload = {
    "features": dummy_features
}

# 4. Send the POST request to your running FastAPI server
print(f"Sending POST request to {URL}...")
try:
    response = requests.post(URL, json=payload)
    
    # 5. Print the results
    print(f"\n[Server Response Code]: {response.status_code}")
    print("[Response Body]:")
    print(response.json())

except requests.exceptions.ConnectionError:
    print("\n❌ Error: Could not connect to the server.")
    print("Make sure your FastAPI server is running on http://127.0.0.1:8001")