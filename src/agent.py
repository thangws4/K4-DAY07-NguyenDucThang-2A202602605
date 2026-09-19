from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    NO_CONTEXT_ANSWER = "I could not find anything about that in the knowledge base."

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return self.NO_CONTEXT_ANSWER

        prompt = self._build_prompt(question, results)
        return self.llm_fn(prompt)

    def _build_prompt(self, question: str, results: list[dict]) -> str:
        """Lay out the retrieved chunks as numbered, attributed context blocks."""
        blocks = []
        for index, result in enumerate(results, start=1):
            source = result["metadata"].get("source_url") or result["metadata"].get("doc_id", "unknown")
            blocks.append(f"[{index}] (source: {source}, score: {result['score']:.3f})\n{result['content']}")
        context = "\n\n".join(blocks)

        return (
            "Answer the question using ONLY the context below.\n"
            "If the context does not contain the answer, say so instead of guessing.\n"
            "Cite the context block number you used, e.g. [1].\n\n"
            f"=== CONTEXT ===\n{context}\n\n"
            f"=== QUESTION ===\n{question}\n\n"
            "=== ANSWER ==="
        )
