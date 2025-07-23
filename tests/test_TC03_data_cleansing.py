import requests
import json
import random
import string
import time

def generate_large_anomalous_dataset(size=100000):
    """
    Function to generate a large dataset with random anomalies for testing.
    - Randomly include missing values, strings, outliers, and valid entries.
    """
    dataset = []
    for i in range(size):
        anomaly_type = random.choice(['missing', 'string', 'outlier', 'valid'])
        if anomaly_type == 'missing':
            dataset.append({"id": i, "value": None})
        elif anomaly_type == 'string':
            dataset.append({"id": i, "value": ''.join(random.choices(string.digits, k=3))})
        elif anomaly_type == 'outlier':
            dataset.append({"id": i, "value": random.randint(1001, 5000)})
        else:
            dataset.append({"id": i, "value": random.randint(0, 1000)})
    return dataset

def test_data_cleansing_and_validation():
    # Generate a large dataset with anomalies
    anomalous_data = generate_large_anomalous_dataset(size=100000)  # Testing with 100,000 records

    try:
        start_time = time.time()

        # Send the anomalous data to the server
        response = requests.post(
            "http://localhost:8080/process_data_with_anomalies", 
            json={"data": anomalous_data}
        )

        # Debugging prints
        print(f"Sent data size: {len(anomalous_data)}")
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.content.decode('utf-8')}")

        # Check the results
        json_response = response.json()

        assert response.status_code == 200, "Failed to process data with anomalies"
        assert json_response['status'] == 'Pass', f"Anomaly correction rate was too low: {json_response['correction_rate']}%"

        print("Test TC03: PASS")
        print(f"Processing time for {len(anomalous_data)} records: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"Test TC03: FAIL - {str(e)}")

if __name__ == "__main__":
    test_data_cleansing_and_validation()
    print("Test TC03 completed.")