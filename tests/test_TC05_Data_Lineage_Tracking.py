import requests

def test_data_lineage_tracking():
    # Step 1: Prepare test data
    pdf_files = ['pdfs/original.pdf', 'pdfs/Coca Cola.pdf']
    files = [('pdf', (pdf, open(pdf, 'rb'))) for pdf in pdf_files]
    
    # Step 2: Send request
    response = requests.post("http://localhost:8080/process_pdf", files=files, data={'question': 'Sample question'})
    
    # Step 3: Verify response
    assert response.status_code == 200, f"Expected status 200 but got {response.status_code}"
    json_response = response.json()
    
    # Step 4: Validate data lineage
    data_lineage = json_response.get('data_lineage', [])
    assert len(data_lineage) > 0, "Data lineage report is missing"
    
    ingestion_steps = [step for step in data_lineage if step['step'] == 'Ingestion']
    assert len(ingestion_steps) == len(pdf_files), "Not all files were ingested"
    
    transformation_step = next(step for step in data_lineage if step['step'] == 'Transformation')
    assert transformation_step['num_chunks'] > 0, "Transformation step did not process correctly"
    
    output_step = next(step for step in data_lineage if step['step'] == 'Final Output')
    assert output_step['response_length'] > 0, "Final output step is missing"
    
    # Final validation
    print("Test TC05: PASS - Data lineage tracked correctly.")
    
if __name__ == "__main__":
    test_data_lineage_tracking()
    print("Test TC05 completed.")