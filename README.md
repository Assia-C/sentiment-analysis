# Sentiment Analysis Classifier API for Book Reviews

This project implements a sentiment analysis classifier to determine the sentiment (positive, negative or neutral) of book reviews from the 'Books_10k.jsonl' dataset. It uses NLP techniques, including text embedding generation with Sentence Transformers and a tuned and trained XGBoost classifier. This project also includes a RESTful API built with FastAPI for making predictions on new text and is using Docker for containerising and for easy setup and deployment.

## Key Features
* **Sentiment Prediction:** Classifies reviews into negative, neutral or positive sentiment.
* **Text Embeddings:** Leverages Sentence Transformers to generate dense vector representations of text.
* **XGBoost Classifier:** Employs a high-performance gradient boosting algorithm for sentiment classification.
* **RESTful API:** Provides an API endpoint for sentiment prediction.
* **Dockerised:** Containerised using Docker for consistent and portable deployment.
* **Unit Tests:** Contains unit tests in the 'tests/' directory for key components of the project.

## Installation

1. **Clone the repository:**
    'git clone https://github.com/Assia-C/sentiment-analysis.git'
2. **Create a virtual environment (recommended):**
    'python -m venv venv'
3. **Install the required dependencies:**
    'pip install -r requirements.txt'

## Usage

### 1. Running the Training Pipeline

**run 'python main.py' to:**
* Load and preprocess the data from Books_10k.jsonl
* Generate text embeddings using the model specified in the config.py
* Train an XGBoost classifier
* Evaluate the model\'s performance and save metrics to models/model_performance.json
* Save the trained model and label encoder

### 2. API

Run the api with:
```bash
'uvicorn api:app --host 0.0.0.0 --port 8000 --reload'
```

Once the API is running, you can send POST requests to the /predict endpoint with a JSON payload containing a list of texts to analyze. For example: 

```bash
curl -X POST -H "Content-Type: application/json" -d '{"texts": ["This is a fantastic book!", "I really disliked this story.", "It was an okay read."]}' http://127.0.0.1:8000/predict
```

### 3. Docker

Build: 'docker build -t sentiment-api .'
Run: 'docker run -p 8000:8000 sentiment-api'

## Testing

Run tests in the 'tests/' directory with 'pytest tests/'. Includes tests for API, data cleaning, and data processing.

## Configuration

See 'config.py' for file paths and embedding model.

## Data

'Books_10k.jsonl': 10,000 book reviews for training. Ratings are mapped to sentiments.

## Docker Setup

'Dockerfile' creates a Python environment and runs the API. '.dockerignore' excludes unnecesary files.

## Future Enhancements

* Address current underperformance of the neutral sentiment category
* Explore and compare different text embedding models
* Investigate alternative sentiment analysis/categorisation models beyond XGBoost
* Implement rate limiting on the API to prevent abuse
* Add input validation to ensure data integrity
* Implement logging for API requests and responses for monitoring and debugging
* Consider adding authentication to the API to control access
* Return confidence scores along with the sentiment predictions

## Author

Assia C.

