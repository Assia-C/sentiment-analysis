import json
import pandas as pd
from data_cleaning import clean_text


def map_rating_to_sentiment(rating):
    """
    Maps a numerical rating to a sentiment label.
    Ratings:
        1&2 -> 'negative'
        3   -> 'neutral'
        4&5 -> 'positive'
    """
    if rating <= 2:
        return "negative"
    elif rating >= 4:
        return "positive"
    else:
        return "neutral"


def parse_jsonl_to_df(file_path):
    """
    Parses a JSONL file containing reviews into a cleaned DataFrame.
    Expects each line to be a JSON object with 'rating' and 'text'.
    Applies sentiment mapping and text cleaning.
    """

    # Initialise lists to store data
    ratings = []
    texts = []
    sentiments = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object
            review = json.loads(line.strip())
            # Extract the rating and skip if missing
            rating = review.get('rating')
            if rating is None:
                continue
            # Extract and review text
            review_text = clean_text(review.get('text', ''))
            # Skip empty or invalid texts
            if review_text is not None and len(review_text) > 1:
                ratings.append(rating)
                texts.append(review_text)
                sentiments.append(map_rating_to_sentiment(rating))

    # Create the DataFrame
    sentiment_df = pd.DataFrame({
        'rating': ratings,
        'text': texts,
        'sentiment': sentiments
    })

    return sentiment_df
