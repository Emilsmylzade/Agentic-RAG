"""
answer.py — Adapter that bridges the Agentic RAG system to the evaluator.

Your evaluator (eval.py) imports two functions:
    from answer import answer_question, fetch_context

This file provides those two functions with the exact same signatures
as your pro_implementation/answer.py, but powered by the agentic RAG.
"""

import json
import asyncio
import nest_asyncio

from openai import OpenAI

nest_asyncio.apply()
from pydantic import BaseModel, Field
from agents import Runner

import chromadb

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    AGENT_MODEL,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    FINAL_K,
    UTILITY_MODEL,
)
from agent import rag_agent


# ── Constants the evaluator dashboard imports ───────────────────────
MODEL = AGENT_MODEL
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL

# ── Shared clients ──────────────────────────────────────────────────
openai_client = OpenAI()
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = chroma_client.get_collection(name=CHROMA_COLLECTION)


# ── Result class matching what your evaluator expects ───────────────
# eval.py does doc.page_content.lower() — so we need this attribute.

class Result(BaseModel):
    page_content: str
    metadata: dict


# ── Internal helpers (same techniques the agent uses) ───────────────

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="Chunk IDs ordered most relevant to least relevant"
    )


def _vector_search(query: str, k: int = RETRIEVAL_K) -> list[Result]:
    """Run a vector search and return Result objects."""
    embedding = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=[query]
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=doc, metadata=meta))
    return chunks


def _rewrite_query(question: str) -> str:
    """Rewrite the question for better retrieval."""
    from litellm import completion
    message = (
        f"You are about to search a company knowledge base. "
        f"Rewrite this question as a short, precise search query:\n\n{question}\n\n"
        f"Respond ONLY with the query, nothing else."
    )
    try:
        response = completion(
            model=f"openai/{UTILITY_MODEL}",
            messages=[{"role": "user", "content": message}],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return question


def _rerank(question: str, chunks: list[Result], top_k: int = FINAL_K) -> list[Result]:
    """Rerank chunks using the utility model."""
    from litellm import completion

    system_prompt = (
        "You are a document re-ranker. Rank the provided chunks by relevance "
        "to the question. Most relevant first. Reply only with the list of "
        "ranked chunk IDs (1-indexed)."
    )

    user_prompt = f"Question: {question}\n\nChunks:\n\n"
    for i, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {i + 1}:\n\n{chunk.page_content[:500]}\n\n"
    user_prompt += "Reply only with the ranked chunk IDs."

    try:
        response = completion(
            model=f"openai/{UTILITY_MODEL}",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=RankOrder,
        )
        order = RankOrder.model_validate_json(
            response.choices[0].message.content
        ).order
        reranked = []
        for idx in order:
            if 1 <= idx <= len(chunks) and len(reranked) < top_k:
                reranked.append(chunks[idx - 1])
        return reranked
    except Exception:
        return chunks[:top_k]


# ═══════════════════════════════════════════════════════════════════
# fetch_context — for retrieval metrics (MRR, nDCG, keyword coverage)
# ═══════════════════════════════════════════════════════════════════

def fetch_context(question: str) -> list[Result]:
    """
    Retrieve relevant chunks for a question.
    Uses query expansion + rerank, same as the agent does internally.
    Returns Result objects with .page_content for keyword checking.
    """
    rewritten = _rewrite_query(question)

    chunks1 = _vector_search(question)
    chunks2 = _vector_search(rewritten)

    # Merge, removing duplicates
    seen = set()
    merged = []
    for chunk in chunks1 + chunks2:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            merged.append(chunk)

    reranked = _rerank(question, merged, FINAL_K)
    return reranked


# ═══════════════════════════════════════════════════════════════════
# answer_question — for answer quality (accuracy, completeness, relevance)
# Runs the FULL AGENT with all tools and self-evaluation.
# ═══════════════════════════════════════════════════════════════════

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Result]]:
    """
    Answer a question using the agentic RAG system.
    Returns (answer_string, retrieved_chunks_list).
    """
    messages = [{"role": "user", "content": question}]
    result = asyncio.run(Runner.run(rag_agent, input=messages))
    answer = result.final_output

    # Fetch context separately so the evaluator has chunks for display
    chunks = fetch_context(question)

    return answer, chunks