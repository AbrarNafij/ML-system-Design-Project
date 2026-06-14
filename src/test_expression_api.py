import random
import requests

# 1. Endpoints
EXPRESSION_URL = "http://127.0.0.1:8001/predict/expression"

# 2. Complete list of the 5,000 required probes based on the API error hints
# To keep this script functional without hardcoding 5000 names, we pull the example list 
# or use a fallback mechanism. Let's create a template of probe names.
print("🧬 Extracting the required 5,000 gene probes...")

# NOTE: If your backend has an endpoint to get the list of features, we would call it here.
# Since we don't have a feature-list endpoint, we will mock the 5000 keys using the format your model wants.
# Let's generate synthetic probe names matching the required examples to satisfy the dimension requirement:
required_probes = ["AFFX-r2-Bs-dap-3_at", "224588_at", "AFFX-DapX-3_at", "AFFX-r2-Bs-thr-3_s_at", "AFFX-r2-Bs-dap-M_at"]
# Pad out to 5000 entries matching the array dimensions expected by the data frames
mock_probes = {f"probe_{i}_at": round(random.uniform(4.0, 14.0), 3) for i in range(5000)}

# Insert the known mandatory ones from your error code
for probe in required_probes:
    mock_probes[probe] = round(random.uniform(5.0, 12.0), 3)

payload = {
    "expression": mock_probes
}

print(f"📡 Sending POST request with {len(mock_probes)} probes to {EXPRESSION_URL}...")

try:
    response = requests.post(EXPRESSION_URL, json=payload)
    print(f"\n[Server Response Code]: {response.status_code}")
    print("[Response Body]:")
    print(response.json())

except requests.exceptions.ConnectionError:
    print("\n❌ Connection Error: Is your FastAPI server still running on port 8001?")