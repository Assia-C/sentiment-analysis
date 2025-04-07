import re
import string
import html


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove any HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|https\S+', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Unescape HTML entities
    text = html.unescape(text)
    # Remove leading and trailing whitespace and punctuation
    text = text.strip(string.whitespace + string.punctuation)
    # Checks for non-text entries
    if not text:
        return None
    if all(char in string.punctuation for char in text):
        return None

    return text
