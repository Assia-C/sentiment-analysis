import pickle
import os
import logging
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import config
from data_processing import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# --- Configuration ---
MODEL_PATH = config.XGBOOST_MODEL_PATH
LABEL_ENCODER_PATH = config.LABEL_ENCODER_PATH
EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME

# --- Load Dependencies ---
model_loaded = False
label_encoder_loaded = False
embedding_model_loaded = False

# Try loading the XGBoost model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
    model_loaded = True
except FileNotFoundError as e:
    logging.error(f"Error loading model: {e}")

# Try loading the embedding model
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logging.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
    embedding_model_loaded = True
except Exception as e:
    logging.error(f"Error loading embedding model: {e}")

# Try loading the label encoder
try:
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    logging.info(f"LabelEncoder loaded successfully from {LABEL_ENCODER_PATH}")
    label_encoder_loaded = True
except FileNotFoundError as e:
    logging.error(f"Error loading LabelEncoder: {e}")


# Define the input request model for a batch of texts
class SentimentRequest(BaseModel):
    texts: List[str]

# Define single result format
class SentimentItem(BaseModel):
    text: str
    sentiment: str


# Define the full response format
class SentimentResponse(BaseModel):
    results: List[SentimentItem]


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    texts = request.texts
    if not texts:
        return {"results": []}
    try:
        # Clean and validate each text
        cleaned_pairs = [(text, clean_text(text)) for text in texts]
        valid_pairs = [(orig, clean) for orig, clean in cleaned_pairs if clean]

        if not valid_pairs:
            return {"results": []}
        
        original_texts, cleaned_texts = zip(*valid_pairs)

        # Convert input texts into sentence embeddings and predict
        embeddings = embedding_model.encode(cleaned_texts)
        predictions = model.predict(embeddings)
        # Decode numeric predictions back to sentiment labels
        decoded_predictions = label_encoder.inverse_transform(predictions).tolist()

        # Construct a response with input text and its predicted sentiment
        results = []
        for text, sentiment in zip(original_texts, decoded_predictions):
            logging.info(f"Predicted sentiment '{sentiment}' for text: '{text}'")
            results.append({"text": text, "sentiment": sentiment})

        return {"results": results}
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing sentiment analysis")

@app.get("/health")
async def health_check():
    # Health check endpoint to verify if all components are loaded
    status = {"status": "OK",
              "model_loaded": model_loaded,
              "label_encoder_loaded": label_encoder_loaded,
              "embedding_model_loaded": embedding_model_loaded}
    # If any component fails to load, return degraded status
    if not all([model_loaded, label_encoder_loaded, embedding_model_loaded]):
        status["status"] = "DEGRADED"
    return status
