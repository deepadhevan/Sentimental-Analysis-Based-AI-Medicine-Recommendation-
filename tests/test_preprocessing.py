from src.preprocessing.text_cleaning import clean_text, normalize_aspect

def test_clean_text():
    assert clean_text("  hello   world ") == "hello world"

def test_normalize_aspect():
    assert normalize_aspect("Side Effect") == "side_effect"
