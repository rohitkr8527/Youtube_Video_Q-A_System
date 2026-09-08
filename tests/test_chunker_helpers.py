from app.ingestion.chunker import approx_tokens, topic_hint


def test_approx_tokens_positive():
    assert approx_tokens("hello world") >= 2


def test_topic_hint_ignores_common_words():
    hint = topic_hint("Hybrid retrieval combines semantic retrieval with keyword retrieval and improves search retrieval.")
    assert "retrieval" in hint
