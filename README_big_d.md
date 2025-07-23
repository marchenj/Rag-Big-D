# Big-D: Document Insight Tool

Big-D is a Flask-based AI assistant that allows users to upload PDF documents, analyze them using OpenAI's GPT-4o model, and generate meaningful insights. It supports PDF text extraction, visualization (word clouds, 3D token plots), and sentiment analysis.

---

## Project Structure

```
rag2/
├── dav.py                    # Main script to run the Flask server and process documents
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── static/                   # Folder for generated visualizations (word cloud, sentiment pie, etc.)
│   ├── wordcloud.png
│   ├── 3d_visualization.html
│   └── sentiment_analysis.png
├── templates/
│   └── index.html            # HTML frontend
├── pdfs/                     # Folder with test PDF files
│   ├── NSF 24-589 - Computer and Information Science and Engineering Core Programs.pdf
│   └── Coca Cola.pdf
└── venv/                     # Virtual environment (usually excluded from version control)
```

---

## Getting Started

### Step 0: Add Your OpenAI API Key

Edit `dav.py`:

```python
api_key = "sk-..."  # Replace with your actual key
```


### Step 1: Create and Activate Virtual Environment

```bash
# Remove old virtual environment (optional)
rm -rf venv

# Create a new virtual environment
python3 -m venv venv

# Activate the environment (Mac/Linux)
source venv/bin/activate

# Activate the environment (Windows)
venv\Scripts\activate
```

### Step 2: Check Python Version

```bash
python3 --version
# Example output:
# Python 3.13.3
```

### Step 3: Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Step 4: Run the Application

```bash
python3 dav.py
```

The app will start running on `http://localhost:8080`

---

## Testing the Application

Upload any of the following PDFs and ask the associated questions:

### Test PDF 1: NSF 24-589 - Computer and Information Science and Engineering Core Programs.pdf (26 pages)
Location: `pdfs/NSF 24-589 - Computer and Information Science and Engineering Core Programs.pdf`

Try these questions:
- Which programs within the Division of Information and Intelligent Systems (IIS) support projects focused on human-centered computing, and what are the program goals?
- Who is the Cognizant Program Officer for the OAC Core Research program as mentioned in the document?
- What types of projects are funded under the NSF CISE Core Programs, and what are their respective budget and duration limits?
- What is the role of Juan J. Li in the NSF 24-589 solicitation, and what is the program that she is responsible for?

### Test PDF 2: Coca Cola.pdf (116 pages)
Location: `pdfs/Coca Cola.pdf`

Try these questions:
- What are the primary strategic issues identified in the strategic analysis of Coca-Cola?
- How does Coca-Cola’s product strategy differ from its competitors, particularly PepsiCo?
- What are the major external factors influencing Coca-Cola’s strategy according to Porter’s Five Forces analysis?
- How does the geographic coverage of Coca-Cola contribute to its global success?


---

## Features

- Semantic PDF analysis
- OpenAI-powered Q&A
- 3D & static graph visualizations
- Sentiment analysis & word clouds
- Ready for deployment on Vercel or Heroku

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

Thanks to the team and contributors for supporting the development of Big D.
