
from opensearchpy import OpenSearch


class OpenSearchStore:
    """Store and retrieve document embeddings using OpenSearch."""

    def __init__(
        self,
        host="localhost",
        port=9200,
        index_name="multi_source_documents"
    ):
        self.index_name = index_name

        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            use_ssl=False,
            verify_certs=False
        )

        if not self.client.ping():
            raise ConnectionError("Could not connect to OpenSearch.")

        self.create_index()

    def create_index(self):
        """Create the vector index if it does not exist."""

        if self.client.indices.exists(index=self.index_name):
            return

        index_body = {
            "settings": {
                "index": {
                    "knn": True
                }
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "source": {"type": "keyword"},
                    "source_type": {"type": "keyword"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    }
                }
            }
        }

        self.client.indices.create(
            index=self.index_name,
            body=index_body
        )

    def add_documents(self, documents, embeddings):
        """Insert document chunks and their vectors."""

        if len(documents) != len(embeddings):
            raise ValueError(
                "Documents and embeddings must have equal lengths."
            )

        for document, embedding in zip(documents, embeddings):
            body = {
                "content": document["content"],
                "source": document["source"],
                "source_type": document["source_type"],
                "embedding": embedding
            }

            self.client.index(
                index=self.index_name,
                body=body,
                refresh="wait_for"
            )

    def search(self, query_embedding, top_k=5, source_types=None):
        """Retrieve relevant chunks and remove duplicate chunks."""

        if top_k < 1:
            return []

        candidate_k = top_k * 3

        search_query = {
            "size": candidate_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": candidate_k
                    }
                }
            }
        }

        if source_types:
            search_query["query"] = {
                "bool": {
                    "must": [search_query["query"]],
                    "filter": [
                        {"terms": {"source_type": source_types}}
                    ]
                }
            }

        response = self.client.search(
            index=self.index_name,
            body=search_query
        )

        results = []
        seen_chunks = set()

        for hit in response["hits"]["hits"]:
            result = dict(hit["_source"])

            normalized_content = " ".join(
                result.get("content", "").lower().split()
            )

            duplicate_key = (
                normalized_content,
                result.get("source", "")
            )

            if duplicate_key in seen_chunks:
                continue

            seen_chunks.add(duplicate_key)
            result["score"] = hit["_score"]
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    def delete_index(self):
        """Delete the index when explicitly requested."""

        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
