

from src.rag_pipeline import RAGPipeline


def test_chunk_text_splits_text():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    text = "one two three four five six"

    chunks = pipeline.chunk_text(
        text,
        chunk_size=3,
        overlap=1
    )

    assert chunks == [
        "one two three",
        "three four five",
        "five six"
    ]


def test_chunk_text_handles_empty_text():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    chunks = pipeline.chunk_text("")

    assert chunks == []


def test_prepare_documents():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    documents = [
        {
            "content": "Python is a programming language.",
            "source": "sample.txt",
            "source_type": "txt"
        }
    ]

    chunks = pipeline.prepare_documents(documents)

    assert len(chunks) == 1
    assert chunks[0]["source"] == "sample.txt"
    assert chunks[0]["source_type"] == "txt"
    assert "Python" in chunks[0]["content"]
