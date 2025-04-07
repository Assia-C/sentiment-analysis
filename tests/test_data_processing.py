import pandas as pd
from data_processing import map_rating_to_sentiment, parse_jsonl_to_df
from unittest.mock import mock_open, patch 


def test_map_rating_to_sentiment_negative():
    assert map_rating_to_sentiment(1) == "negative"
    assert map_rating_to_sentiment(2) == "negative"


def test_map_rating_to_sentiment_neutral():
    assert map_rating_to_sentiment(3) == "neutral"


def test_map_rating_to_sentiment_positive():
    assert map_rating_to_sentiment(4) == "positive"
    assert map_rating_to_sentiment(5) == "positive"


def test_parse_jsonl_to_df_empty_file():
    with patch('builtins.open', mock_open(read_data='')):
        df = parse_jsonl_to_df('dummy_path.jsonl')
        assert isinstance(df, pd.DataFrame)
        assert df.empty


def test_parse_jsonl_to_df_single_review():
    review_data = '{"rating": 5, "text": "This is a great book!"}'
    with patch('builtins.open', mock_open(read_data=review_data)):
        df = parse_jsonl_to_df('dummy_path.jsonl')
        assert not df.empty
        assert len(df) == 1
        assert df['rating'].iloc[0] == 5
        assert df['text'].iloc[0] == "this is a great book"
        assert df['sentiment'].iloc[0] == "positive"


def test_parse_jsonl_to_df_multiple_reviews_and_sentences():
    review_data = '{"rating": 1, "text": "Awful book. Really bad."}\n{"rating": 4, "text": "Good read. Enjoyed it a lot."}'
    with patch('builtins.open', mock_open(read_data=review_data)):
        df = parse_jsonl_to_df('dummy_path.jsonl')
        assert len(df) == 2
        assert all(s == "negative" for s in df[df['rating'] == 1]['sentiment'])
        assert all(s == "positive" for s in df[df['rating'] == 4]['sentiment'])


def test_parse_jsonl_to_df_skips_short_sentences():
    review_data = '{"rating": 3, "text": "Ok. It was fine."}'
    with patch('builtins.open', mock_open(read_data=review_data)):
        df = parse_jsonl_to_df('dummy_path.jsonl')
        assert len(df) == 1


def test_parse_jsonl_to_df_handles_missing_text():
    review_data = '{"rating": 4}'
    with patch('builtins.open', mock_open(read_data=review_data)):
        df = parse_jsonl_to_df('dummy_path.jsonl')
        assert df.empty  # Should not create rows if no text


def test_parse_jsonl_to_df_handles_missing_rating():
    review_data = '{"text": "Good book."}'
    with patch('builtins.open', mock_open(read_data=review_data)):
        df = parse_jsonl_to_df('dummy_path.jsonl')
        assert df.empty  # Should not create rows if no rating
