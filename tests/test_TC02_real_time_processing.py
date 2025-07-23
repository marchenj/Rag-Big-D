import requests
import time
from PyPDF2 import PdfReader

def split_pdf_into_chunks(pdf_path, chunk_size=100):
    reader = PdfReader(pdf_path)
    text_chunks = []
    full_text = ""
    
    for page in reader.pages:
        full_text += page.extract_text()

    # Split the full text into chunks
    words = full_text.split()
    for i in range(0, len(words), chunk_size):
        text_chunks.append(' '.join(words[i:i+chunk_size]))
    
    return text_chunks

def test_real_time_data_processing():
    # Split the PDF into chunks
    pdf_path = 'pdfs/Python for Finance Analyze Big Financial Data.pdf'
    real_time_data_chunks = split_pdf_into_chunks(pdf_path)

    try:
        for i, data_chunk in enumerate(real_time_data_chunks):
            start_time = time.time()
            
            # Send the data chunk to the server
            response = requests.post("http://localhost:8080/process_real_time", json={"data": data_chunk})
            
            # Record the response time
            elapsed_time = time.time() - start_time
            
            # Debugging prints
            print(f"Sent data chunk {i + 1}: {data_chunk[:100000]}...")  # Print only the first 100000 chars of the chunk
            print(f"Response status code: {response.status_code}")
            print(f"Response content: {response.content}")
            print(f"Time taken for chunk {i + 1}: {elapsed_time:.4f} seconds\n")

            # Assert that the processing time is under 2 seconds
            assert response.status_code == 200, f"Failed to process real-time data, status code: {response.status_code}"
            assert elapsed_time < 2, f"Latency exceeded 2 seconds for chunk {i + 1}. Processing time: {elapsed_time:.4f} seconds"
        
        print("Test TC02: PASS")

    except Exception as e:
        print(f"Test TC02: FAIL - {str(e)}")

if __name__ == "__main__":
    test_real_time_data_processing()
    print("Test TC02 completed.")