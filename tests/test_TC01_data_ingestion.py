import requests
import os

def test_data_ingestion():
    # Step 1: Prepare Data
    pdf_files = [ 'pdfs/Python for Finance Analyze Big Financial Data.pdf']  
    open_files = [open(pdf, 'rb') for pdf in pdf_files]
    files = [('pdf', (os.path.basename(pdf), f)) for pdf, f in zip(pdf_files, open_files)]

    try:
        # Step 2: Send Request to the /process_pdf endpoint
        response = requests.post(
            "http://localhost:8080/process_pdf", 
            files=files, 
            data={'question': 'Sample question'}
        )

        # Debugging prints
        print(f"Request to /process_pdf with files: {pdf_files}")
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.content}")

        # Step 3: Check Results
        assert response.status_code == 200, f"Failed to ingest data, HTTP error occurred with status code {response.status_code}"
        
        json_response = response.json()
        response_text = json_response['response']

        # Check if the response indicates successful ingestion
        if "Files successfully ingested and stored." in response_text:
            print("Test TC01: PASS")
        else:
            print("Test TC01: FAIL - Unexpected response")
            print(f"Response was: {response_text}")

    finally:
        # Ensure all files are properly closed
        for f in open_files:
            f.close()

if __name__ == "__main__":
    test_data_ingestion()
    print("Test TC01 completed.")