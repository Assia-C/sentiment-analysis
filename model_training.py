from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import numpy as np
import pickle
import os
import time
import json
import config
from data_processing import parse_jsonl_to_df
from embedding_generation import prepare_data_for_training


def train_xgboost_model(X_train, y_train, save_path=None):
    """
    Trains an XGBoost classifier using the provided training data and saves 
    the model to disk.
    """
    print("Training XGBoost model...")
    start_time = time.time()

    best_params = {
        'colsample_bytree': 1.0,
        'learning_rate': 0.2,
        'max_depth': 7,
        'n_estimators': 300,
        'subsample': 1.0
    }

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        eval_metric='merror',
        random_state=42,
        **best_params
    )

    model.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    print(f"XGBoost model trained in {elapsed_time:.2f} seconds")

    # Save model to disk
    save_path = config.XGBOOST_MODEL_PATH if save_path is None else save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"XGBoost model saved to {save_path}")
    return model


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluates the trained model on the test data and prints performance metrics.
    Also measures and logs preediction latency.
    """
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Print confusion matrix
    print("Confustion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Measure prediction latency (P99)
    print("Measuring prediction latency...")
    latencies = []
    for _ in range(100):
        sample_idx = np.random.choice(len(X_test), size=10)
        X_sample = X_test[sample_idx]

        start_time = time.time()
        _ = model.predict(X_sample)
        latency = (time.time() - start_time) * 1000/ len(X_sample)
        latencies.append(latency)

    p99_latency = np.percentile(latencies, 99)
    print(f"P99 latency: {p99_latency:.2f} ms per prediction")

    # Save metrics to file
    performance_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": report,
        "confusion_matrix": cm.tolist() if isinstance(cm, np.ndarray) else cm,
        "p99_latency_ms_per_prediction": p99_latency
    }

    os.makedirs(os.path.dirname(config.MODEL_PERFORMANCE_PATH), exist_ok=True)
    with open(config.MODEL_PERFORMANCE_PATH, "w") as f:
        json.dump(performance_metrics, f, indent=4)

    print(f"Model performance metrics and latency saved to {config.MODEL_PERFORMANCE_PATH}")

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "p99_latency": p99_latency
    }


def run_training_pipeline(jsonl_file, model_name=None, label_encoder_save_path=None):
    """
    Runs the full training pipeline:
    - Loads and parses input data
    - Encodes labels
    - Applies SMOTE for class balancing
    - Trains and evaluates an XGBoost classifier
    """
    print(f"Loading data from {jsonl_file}...")
    sentiment_df = parse_jsonl_to_df(jsonl_file)

    model_name = config.EMBEDDING_MODEL_NAME if model_name is None else model_name

    # Generate embeddings foir train and test sets
    X_train, X_test, y_train_original, y_test_original, train_df, test_df = prepare_data_for_training(
        sentiment_df,
        embedding_model=model_name
    )

    # Encode str labels to integers (0=negative, 1=neutral, 2=positive)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_original)
    y_test = label_encoder.transform(y_test_original)

    # Save fitted label encoder
    label_encoder_save_path = config.LABEL_ENCODER_PATH if label_encoder_save_path is None else label_encoder_save_path
    os.makedirs(os.path.dirname(label_encoder_save_path), exist_ok=True)
    with open(label_encoder_save_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"LabelEncoder saved to {label_encoder_save_path}")


    # Use SMOTE to handle class imbalance in training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Using XGBoost with tuned hyperparameters for training...")
    model = train_xgboost_model(X_train_resampled, y_train_resampled)

    # Evaluate and log performance
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        class_names=['negative', 'neutral', 'positive']
    )

    return model, metrics, train_df, test_df
