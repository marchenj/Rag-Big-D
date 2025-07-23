import time
import openai
import os
import tempfile
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI environments
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI  


# Download NLTK data if not already present
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Set up Flask app
app = Flask(__name__)

# Set your OpenAI API key and assistant ID
api_key = "sk-proj-..."
os.environ["OPENAI_API_KEY"] = api_key  # Set OpenAI API key for use with langchain
openai.api_key = api_key  # Set OpenAI API key for use with openai library

# Initialize OpenAI client 
client = OpenAI(api_key=api_key)

# Initialize list to store processing times
processing_times = []

# Function to extract text from a single PDF file using PyPDFLoader
def extract_text_pypdf_loader(file):
  try:
      # Save the file temporarily
      with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
          file.save(temp_file.name)
          temp_file_path = temp_file.name
    
      #Using PyPDFLoader with temporary file path
      loader = PyPDFLoader(temp_file_path)
      pages = loader.load()  # This returns a list of Documents, one per page
      text = ""
      for page in pages:
          if page.page_content:  # Ensure the page content is not None
              text += page.page_content

      # Delete temporary file after upload
      os.remove(temp_file_path)

      return text
  except Exception as e:
      print(f"Error reading PDF with PyPDFLoader: {e}")
      return None  # Return None to indicate failure

# POINT 1

# Function to preprocess text (remove stopwords, lemmatize, etc.)
def preprocess_text(text):
  text = text.lower()
  text = re.sub(r'[^\w\s]', '', text)
  words = word_tokenize(text)
  stop_words = set(stopwords.words('english'))
  filtered_words = [WordNetLemmatizer().lemmatize(word) for word in words if word not in stop_words]
  return ' '.join(filtered_words)

# Function to generate a word cloud
def generate_word_cloud(text):
  wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
  plt.figure(figsize=(10, 5))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.savefig('static/wordcloud.png')  # Save the word cloud as an image
  plt.close()

# Function to create 3D visualization of tokenization
def plot_3d_tokenization(text):
  tokens = text.split()
  embeddings = np.array([np.random.rand(300) for _ in tokens])
  pca = PCA(n_components=3)
  reduced_embeddings = pca.fit_transform(embeddings)
  trace = go.Scatter3d(
      x=reduced_embeddings[:, 0],
      y=reduced_embeddings[:, 1],
      z=reduced_embeddings[:, 2],
      mode='markers+text',
      text=tokens,
      marker=dict(
          size=5,
          color=reduced_embeddings[:, 0],
          colorscale='Viridis',
          opacity=0.8
      )
  )
  layout = go.Layout(
      title='3D Visualization of Tokenization',
      margin=dict(l=0, r=0, b=0, t=0)
  )
  fig = go.Figure(data=[trace], layout=layout)
  fig.write_html('static/3d_visualization.html')

# Function to perform sentiment analysis
def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [sentiment['pos'], sentiment['neu'], sentiment['neg']]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Sentiment Analysis')
    plt.savefig('static/sentiment_analysis.png')
    plt.close()


# Function to update the processing time graph
def update_processing_time_graph(times):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(times)), times, marker='o', linestyle='-', color='b')
    plt.xlabel('Request Number')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time per Request')
    plt.savefig('static/processing_time.png')  # Save the processing time graph as an image
    plt.close()

###############
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
  print("Received files:", request.files)
  print("Received form data:", request.form)
  start_time = time.time()
  uploaded_files = request.files.getlist('pdf')
  question = request.form.get('question', '')

  if not uploaded_files:
      return jsonify({"response": "No files uploaded.", "processing_time": time.time() - start_time}), 400

  extracted_texts = []
  for uploaded_file in uploaded_files:
      extracted_text = extract_text_pypdf_loader(uploaded_file)
      if extracted_text is None:
          return jsonify({"response": "Failed to extract text from the provided PDF(s). Please check the file format and content.", "processing_time": time.time() - start_time}), 400
      extracted_texts.append(extracted_text)

  if len(extracted_texts) == 0:
      return jsonify({"response": "No valid data extracted from files.", "processing_time": time.time() - start_time}), 400

  documents = [Document(page_content=text) for text in extracted_texts]
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
  split_texts = text_splitter.split_documents(documents)

# POINT 2 - Updated embeddings usage
  embeddings = OpenAIEmbeddings()  
  doc_texts = [doc.page_content for doc in split_texts]
  doc_embeddings = embeddings.embed_documents(doc_texts)

  # Improved Query Matching
  def retrieve_relevant_documents(query):
      query_embedding = embeddings.embed_query(query)
      similarities = cosine_similarity([query_embedding], doc_embeddings)
      
      # Get top 3 most similar documents
      most_similar_indices = np.argsort(similarities[0])[::-1][:3]
      retrieved_texts = [split_texts[idx].page_content for idx in most_similar_indices]
      return ' '.join(retrieved_texts)  # Combine the top results to provide a richer context


  context = retrieve_relevant_documents(question)

  # Generate visualizations
  full_text = " ".join(extracted_texts)
  processed_text = preprocess_text(full_text)  # Apply text preprocessing
  generate_word_cloud(processed_text)
  plot_3d_tokenization(processed_text)
  sentiment_analysis(processed_text)


# Point 3 - Updated OpenAI API call
  # Generate a response using OpenAI API
  response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {
              "role": "system",
              "content": "You are an assistant who analyzes documents and answers questions based on the content provided."
          },
          {
              "role": "user",
              "content": f"I have a document that I need help analyzing. The context is: {context}. Here is my question: {question}. Please provide relevant insights or information based on the document."
          }
      ]
  )

  # Extract the response from the API 
  response_content = response.choices[0].message.content
  end_time = time.time()
  processing_time = end_time - start_time

  # Append the new processing time to the list
  processing_times.append(processing_time)

  # Update the processing time graph
  update_processing_time_graph(processing_times)

  # Return the response from OpenAI API
  return jsonify({"response": response_content, "processing_time": processing_time})


if __name__ == '__main__':
  from waitress import serve
  serve(app, host='0.0.0.0', port=8080)