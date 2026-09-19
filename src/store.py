from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Backed by a plain in-memory list. The ChromaDB branch that the skeleton
    sketched is deliberately left out: nothing in requirements.txt installs it,
    no test exercises it, and a store whose behaviour depends on what happens to
    be installed would make benchmark scores incomparable between students.

    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Normalize one Document into a stored record (content + metadata + vector)."""
        metadata = dict(doc.metadata or {})
        # delete_document() and metadata filters key off doc_id, so it must always
        # be there. setdefault, not assignment: when several chunks of one file are
        # added as Document("file#0"), Document("file#1"), ... the loader has already
        # put the *file's* id in metadata and that is what should survive.
        metadata.setdefault("doc_id", doc.id)

        record = {
            "id": f"{doc.id}#{self._next_index}",
            "doc_id": metadata["doc_id"],
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Rank the given records against the query and return the best top_k."""
        if top_k <= 0 or not records:
            return []

        query_embedding = self._embedding_fn(query)
        # Embeddings from every backend in src/embeddings.py are L2-normalized,
        # so the dot product equals cosine similarity here. The stored vector is
        # left out of the result: it is noise when a result is printed.
        scored = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": _dot(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        scored.sort(key=lambda result: result["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        One Document becomes one record: chunking happens in the caller, so that
        a chunking strategy can be swapped without touching the store.
        """
        if not docs:
            return
        self._store.extend(self._make_record(doc) for doc in docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        Computes the dot product of the query embedding against every stored
        embedding, then returns the highest scoring records.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        Filters first, then ranks. Ranking first would spend all top_k slots on
        documents that the filter then removes, so a store holding plenty of
        matching documents could still return nothing.
        """
        if not metadata_filter:
            candidates = self._store
        else:
            candidates = [
                record
                for record in self._store
                if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
            ]

        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        remaining = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
        if len(remaining) == len(self._store):
            return False

        self._store = remaining
        return True
