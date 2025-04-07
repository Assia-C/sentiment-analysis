import os
from model_training import run_training_pipeline
import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """Runs the sentiment analysis pipeline."""

    jsonl_file = 'Books_10k.jsonl'

    try:
        logging.info(f"Starting training pipeline with data from: {jsonl_file}")
        model, metrics, _, _ = run_training_pipeline(jsonl_file)
        logging.info("Training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed:{e}")
        exit(1)

    # Print the metrics and verify the trained models
    print("\nTraining Pipeline Results (XGBoost):")
    print("Classification Report:")
    print(metrics['classification_report'])
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nP99 Latency:")
    print(f"{metrics['p99_latency']:.2f} ms")

    # Verify model saving for XGBoost
    if os.path.exists(config.XGBOOST_MODEL_PATH):
        print(f"\nXGBoost model saved successfully at: {config.XGBOOST_MODEL_PATH}")
    else:
        print("\nXGBoost model saving failed.")


if __name__ == '__main__':
    main()
