
import requests

from src.loaders import MultiSourceLoader
from src.embeddings import EmbeddingModel
from src.opensearch_store import OpenSearchStore


class RAGPipeline:
    """Coordinate ingestion, retrieval and answer generation."""

    def __init__(self, ollama_model="llama3.2:3b"):
        self.loader = MultiSourceLoader()
        self.embedding_model = EmbeddingModel()
        self.store = OpenSearchStore()

        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/chat"

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be non-negative and smaller than chunk_size."
            )

        words = text.split()
        chunks = []

        if not words:
            return chunks

        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            if end >= len(words):
                break

            start = end - overlap

        return chunks

    def prepare_documents(self, documents):
        """Split loaded documents into smaller searchable chunks."""

        chunks = []

        for document in documents:

            text_chunks = self.chunk_text(
                document["content"]
            )

            for chunk in text_chunks:

                chunks.append({
                    "content": chunk,
                    "source": document["source"],
                    "source_type": document["source_type"]
                })

        return chunks

    def ingest_documents(self, documents):
        """Generate embeddings and index document chunks."""

        if not documents:
            return 0

        chunks = self.prepare_documents(documents)

        if not chunks:
            return 0

        texts = [
            chunk["content"]
            for chunk in chunks
        ]

        embeddings = self.embedding_model.generate_embeddings(
            texts
        )

        self.store.add_documents(
            chunks,
            embeddings
        )

        return len(chunks)

    def ingest_file(self, file_path):
        """Load and index a CSV, TXT or Markdown file."""

        documents = self.loader.load_file(file_path)

        return self.ingest_documents(documents)

    def ingest_sqlite(self, database_path, table_name):
        """Load and index a SQLite database table."""

        documents = self.loader.load_sqlite(
            database_path,
            table_name
        )

        return self.ingest_documents(documents)

    def retrieve(self, question, top_k=5, source_types=None):
        """Find relevant and non-duplicate document chunks."""

        if not question or not question.strip():
            return []

        if top_k < 1:
            return []

        query_embedding = (
            self.embedding_model.generate_query_embedding(
                question
            )
        )

        retrieved_documents = self.store.search(
            query_embedding,
            top_k=top_k,
            source_types=source_types
        )

        return retrieved_documents

    def generate_answer(self, question, retrieved_documents):
        """Generate an answer using retrieved document context."""

        if not retrieved_documents:
            return (
                "I could not find relevant information "
                "in the selected sources."
            )

        context_parts = []

        for index, document in enumerate(
            retrieved_documents,
            start=1
        ):

            source = document.get(
                "source",
                "Unknown source"
            )

            source_type = document.get(
                "source_type",
                "Unknown"
            )

            content = document.get(
                "content",
                ""
            )

            context_parts.append(
                f"""
[Source {index}]
File: {source}
Source type: {source_type}
Content: {content}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are Aavya AI, a helpful multi-source knowledge assistant.

Answer the user's question using ONLY the supplied context.

IMPORTANT RULES:

1. Never invent facts or information.
2. If the answer is not available in the context,
   clearly say that the information was not found.
3. Give clear, direct and concise answers.
4. Use information from multiple sources when relevant.
5. Do not confuse information from different documents.
6. Cite the supporting source using its exact label,
   for example: [Source 1] or [Source 2].
7. Only cite sources that actually support your answer.
8. If sources provide different information, explain the difference.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.ollama_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        return response.json()["message"]["content"]

    def ask(self, question, top_k=5, source_types=None):
        """Retrieve relevant information and generate an answer."""

        retrieved_documents = self.retrieve(
            question,
            top_k=top_k,
            source_types=source_types
        )

        answer = self.generate_answer(
            question,
            retrieved_documents
        )

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_documents
        }
