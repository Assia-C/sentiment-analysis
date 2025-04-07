
from data_cleaning import clean_text


def test_clean_text_lowercase():
    text = "This IS a Test"
    assert clean_text(text) == "this is a test"


def test_clean_text_removes_html():
    text = "<p>This is <b>bold</b> text.</p>"
    assert clean_text(text) == "this is bold text"


def test_clean_text_removes_urls():
    text = "Check out this site: https://www.example.com and http://anothersite.org"
    assert clean_text(text) == "check out this site: and"


def test_clean_text_removes_multiple_spaces():
    text = "This  has   multiple    spaces."
    assert clean_text(text) == "this has multiple spaces"


def test_clean_text_unescapes_html_entities():
    text = "This has &amp; and &quot; in it."
    assert clean_text(text) == "this has & and \" in it"


def test_clean_text_removes_leading_trailing_whitespace_punctuation():
    text = "  ! Hello?  "
    assert clean_text(text) == "hello"


def test_clean_text_removes_leading_trailing_specific_punctuation():
    text = ",,World!!"
    assert clean_text(text) == "world"


def test_clean_text_does_not_remove_internal_important_punctuation():
    text = "Is this okay? Yes, it is!"
    assert clean_text(text) == "is this okay? yes, it is"


def test_clean_text_returns_none_for_empty_string():
    text = ""
    assert clean_text(text) is None


def test_clean_text_returns_none_for_only_whitespace():
    text = "   "
    assert clean_text(text) is None


def test_clean_text_returns_none_for_only_punctuation():
    text = ",.;:"
    assert clean_text(text) is None


def test_clean_text_returns_none_for_leading_trailing_only_punctuation():
    text = " ,.;: "
    assert clean_text(text) is None
