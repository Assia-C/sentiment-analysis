from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def extract_sentiments(response_json):
    return [item["sentiment"] for item in response_json["results"]]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["OK", "DEGRADED"]
    assert "model_loaded" in data
    assert "label_encoder_loaded" in data
    assert "embedding_model_loaded" in data


def test_predict_endpoint_single_positive():
    response = client.post(
        "/predict",
        json={"texts": ["This is absolutely fantastic!"]},
    )
    assert response.status_code == 200
    sentiments = extract_sentiments(response.json())
    assert sentiments[0] == "positive"


def test_predict_endpoint_single_negative():
    response = client.post(
        "/predict",
        json={"texts": ["This is a truly awful experience."]},
    )
    assert response.status_code == 200
    sentiments = extract_sentiments(response.json())
    assert sentiments[0] == "negative"


def test_predict_endpoint_single_neutral():
    response = client.post(
        "/predict",
        json={"texts": ["The service was okay."]},
    )
    assert response.status_code == 200
    sentiments = extract_sentiments(response.json())
    assert sentiments[0] == "neutral"


def test_predict_endpoint_batch_mixed():
    test_texts = [
        "I loved this!",
        "It was terrible.",
        "This was an okay book.",
    ]
    response = client.post(
        "/predict",
        json={"texts": test_texts},
    )
    assert response.status_code == 200
    sentiments = extract_sentiments(response.json())
    assert len(sentiments) == len(test_texts)
    assert "positive" in sentiments
    assert "negative" in sentiments
    assert "neutral" in sentiments


def test_predict_endpoint_empty_list():
    response = client.post(
        "/predict",
        json={"texts": []},
    )
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_predict_endpoint_with_punctuation():
    response = client.post(
        "/predict",
        json={"texts": ["What a fantastic book!" , "This is so bad..."]},
    )
    assert response.status_code == 200
    sentiments = extract_sentiments(response.json())
    assert sentiments[0] == "positive"
    assert sentiments[1] == "negative"


def test_predict_endpoint_long_text():
    long_text = "This is a very long piece of text that should still be processed correctly by the sentiment analysis model. It contains many words and sentences to simulate a more realistic review scenario. Hoping the model can handle this extended input without any issues."
    response = client.post(
        "/predict",
        json={"texts": [long_text]},
    )
    assert response.status_code == 200
    sentiments = extract_sentiments(response.json())
    assert sentiments[0] in ["positive", "neutral", "negative"]

