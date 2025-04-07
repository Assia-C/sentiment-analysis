from sentence_transformers import SentenceTransformer
import pickle
import os
import time
from sklearn.model_selection import train_test_split
import config


def generate_embeddings(sentences, model_name=None, batch_size=32, cache_file=None):
    """
    Generate sentence embeddings using a SentenceTransformer model.
    Optionally uses caching to speed up repeated runs.
    """
    if model_name is None:
        model_name = config.EMBEDDING_MODEL_NAME

    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Generating embeddings using {model_name}...")
    start_time = time.time()

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed_time = time.time() - start_time
    print(f"Embeddings generated in {elapsed_time:.2f} seconds")
    print(f"Embedding shape: {embeddings.shape}")

    # Save embeddings to cache for future reuse
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings cached to {cache_file}")

    return embeddings


def prepare_data_for_training(sentence_df, embedding_model=None, cache_dir='./cache'):
    """
    Splits data into train/test sets, generates and optionally caches embeddings.
    """
    if embedding_model is None:
        embedding_model = config.EMBEDDING_MODEL_NAME

    os.makedirs(cache_dir, exist_ok=True)

    train_df, test_df = train_test_split(
        sentence_df,
        test_size=0.2,
        random_state=42,
        stratify=sentence_df['sentiment']
    )

    # Generate train embeddings
    train_cache = os.path.join(cache_dir, f'train_embeddings_{embedding_model.replace('/', '_')}.pkl')
    train_embeddings = generate_embeddings(
        train_df['text'].to_list(),
        model_name=embedding_model,
        cache_file=train_cache
    )
    # Generate test embeddings
    test_cache = os.path.join(cache_dir, f'test_embeddings_{embedding_model.replace('/', '_')}.pkl')
    test_embeddings = generate_embeddings(
        test_df['text'].to_list(),
        model_name=embedding_model,
        cache_file=test_cache
    )

    return (
        train_embeddings,
        test_embeddings,
        train_df['sentiment'].values,
        test_df['sentiment'].values,
        train_df,
        test_df
    )
