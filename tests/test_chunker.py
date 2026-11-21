import pytest

from recallengine.chunker import chunk_text


def test_chunk_text_sliding_window_overlap():
    text = "one two three four five six seven eight nine ten eleven twelve"
    chunks = chunk_text(text, chunk_size=6, overlap=2)

    assert chunks == [
        "one two three four five six",
        "five six seven eight nine ten",
        "nine ten eleven twelve",
    ]


def test_chunk_text_shorter_than_window():
    text = "short single chunk text"
    assert chunk_text(text, chunk_size=150, overlap=30) == ["short single chunk text"]


def test_chunk_text_empty_or_whitespace():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_invalid_parameters():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=0, overlap=0)

    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=10, overlap=10)
