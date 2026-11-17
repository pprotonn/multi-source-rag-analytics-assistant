
from unittest.mock import MagicMock

from src.opensearch_store import OpenSearchStore


def create_test_store():
    """Create a store with a mocked OpenSearch client."""

    store = OpenSearchStore.__new__(OpenSearchStore)

    store.index_name = "test_documents"
    store.client = MagicMock()

    return store


def test_search_returns_documents():
    """Check whether search returns relevant documents."""

    store = create_test_store()

    store.client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 0.95,
                    "_source": {
                        "content": "Python is a programming language.",
                        "source": "sample.txt",
                        "source_type": "txt"
                    }
                }
            ]
        }
    }

    results = store.search(
        query_embedding=[0.1] * 384,
        top_k=5
    )

    assert len(results) == 1
    assert results[0]["content"] == "Python is a programming language."
    assert results[0]["source"] == "sample.txt"
    assert results[0]["score"] == 0.95


def test_search_uses_top_k():
    """Check whether search fetches extra candidates for deduplication."""

    store = create_test_store()

    store.client.search.return_value = {
        "hits": {
            "hits": []
        }
    }

    store.search(
        query_embedding=[0.1] * 384,
        top_k=3
    )

    call_arguments = store.client.search.call_args.kwargs
    search_body = call_arguments["body"]

    # top_k is 3, so the system fetches 9 candidates.
    assert search_body["size"] == 9
    assert search_body["query"]["knn"]["embedding"]["k"] == 9


def test_search_filters_source_types():
    """Check whether source-type filtering works."""

    store = create_test_store()

    store.client.search.return_value = {
        "hits": {
            "hits": []
        }
    }

    store.search(
        query_embedding=[0.1] * 384,
        top_k=5,
        source_types=["csv", "txt"]
    )

    call_arguments = store.client.search.call_args.kwargs
    query = call_arguments["body"]["query"]

    assert "bool" in query

    filters = query["bool"]["filter"]

    assert {
        "terms": {
            "source_type": ["csv", "txt"]
        }
    } in filters


def test_search_without_source_filter():
    """Check whether search works without source filtering."""

    store = create_test_store()

    store.client.search.return_value = {
        "hits": {
            "hits": []
        }
    }

    store.search(
        query_embedding=[0.1] * 384,
        top_k=5,
        source_types=None
    )

    call_arguments = store.client.search.call_args.kwargs
    query = call_arguments["body"]["query"]

    assert "knn" in query
