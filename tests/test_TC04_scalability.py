import requests
import time
import json

def generate_data(size):
    """Generate a dataset of the given size."""
    return [{"id": i, "value": i * 10} for i in range(size)]

def test_scalability():
    data_sizes = [10, 100, 1000, 10000]  # Exponentially increasing data sizes
    processing_times = []
    absolute_time_threshold = 0.1  # Set a time threshold of 0.1 seconds for large datasets

    for size in data_sizes:
        data = generate_data(size)
        try:
            print(f"Testing with {size} records...")

            start_time = time.time()
            response = requests.post(
                "http://localhost:8080/process_large_data", 
                json={"data": data}
            )
            end_time = time.time()

            # Debugging information
            print(f"Sent {size} records. Response status code: {response.status_code}")
            print(f"Response content: {response.content.decode('utf-8')}")

            if response.status_code == 200:
                processing_time = end_time - start_time
                print(f"Time taken to process {size} records: {processing_time:.4f} seconds")
                processing_times.append(processing_time)
            else:
                print(f"Failed to process {size} records")

        except Exception as e:
            print(f"Test TC04: FAIL - Error occurred for data size {size}: {str(e)}")
            return

    # Assess performance
    for i in range(1, len(processing_times)):
        degradation = (processing_times[i] - processing_times[i-1]) / processing_times[i-1] * 100
        print(f"Performance degradation from {data_sizes[i-1]} to {data_sizes[i]} records: {degradation:.2f}%")

        # Only check degradation for smaller datasets (below 1000 records)
        if data_sizes[i] <= 1000:
            assert degradation < 60, f"Test TC04: FAIL - Performance degraded by {degradation:.2f}%"

    # Additional check: Ensure processing time stays below the absolute threshold for large datasets
    assert processing_times[-1] < absolute_time_threshold, (
        f"Test TC04: FAIL - Processing time for {data_sizes[-1]} records exceeded {absolute_time_threshold} seconds."
    )

    print("Test TC04: PASS - System scaled effectively")
    print(f"Processing times: {processing_times}")

if __name__ == "__main__":
    test_scalability()
    print("Test TC04 completed.")