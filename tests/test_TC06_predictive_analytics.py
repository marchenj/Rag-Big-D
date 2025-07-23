import requests
import json

def test_predictive_analytics():
    url = "http://localhost:8080/process_pdf"
    pdf_files = ['pdfs/Coca Cola.pdf']  # Make sure this path exists in your project
    
    files = [('pdf', (file, open(file, 'rb'))) for file in pdf_files]
    
    response = requests.post(url, files=files, data={'question': 'Predict future value'})
    
    assert response.status_code == 200, "Failed to process PDF files"
    
    json_response = response.json()
    
    print("Predicted top words:", json_response['predicted_value'])
    print("Response Time:", json_response['processing_time'])
    
    # Update expected words list based on the PDF and word cloud analysis
    expected_words = [
        'Coca Cola', 'Strategy', 'Noncarbonated', 'Carbonated', 'company', 
        'market', 'product', 'brand', 'industry', 'consumer', 'drink', 
        'percent', 'beverage', 'growth', 'PepsiCo', 'bottler', 'competitive'
    ]
    
    common_words = set(json_response['predicted_value']).intersection(set(expected_words))
    
    print(f"Common words found: {common_words}")
    assert len(common_words) >= 5, f"Prediction failed: only {len(common_words)} common words found."
    
    print("Test TC06: PASS - Predictive analytics using word cloud successful.")

if __name__ == "__main__":
    test_predictive_analytics()