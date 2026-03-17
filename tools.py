"""
tools.py — The toolbox for the RAG agent.

Each function here becomes a tool the agent can call.
The agent decides WHICH tools to use and in WHAT ORDER.

Tools:
  1. vector_search     — Semantic similarity search in ChromaDB
  2. keyword_search    — Exact string matching across knowledge base files
  3. list_sources      — See what document types and files are available
  4. rerank_chunks     — Ask an LLM to reorder chunks by relevance
  5. self_evaluate      — Score an answer and get improvement suggestions
"""

import json
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field
from agents import function_tool, RunContextWrapper

import chromadb

from config import (
    KNOWLEDGE_BASE_PATH,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    UTILITY_MODEL,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    FINAL_K,
    MIN_ACCEPTABLE_SCORE,
)

client = OpenAI()


# ── Helper: get the ChromaDB collection ─────────────────────────────

def _get_collection():
    """Connect to the existing ChromaDB collection."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return chroma_client.get_collection(name=CHROMA_COLLECTION)


# ═══════════════════════════════════════════════════════════════════
# Tool 1 — Vector Search (semantic similarity)
# ═══════════════════════════════════════════════════════════════════

@function_tool
def vector_search(query: str, num_results: int = 15) -> str:
    """
    Search the knowledge base using semantic similarity.
    Converts the query into a vector and finds the closest matching chunks.

    Use this as your PRIMARY search tool. Good for natural language questions.

    Args:
        query: The search query (a question or topic to find information about)
        num_results: How many chunks to retrieve (default 15, max 25)
    """
    num_results = min(num_results, 25)
    collection = _get_collection()

    # Turn the query into a vector
    embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL, input=query
    )
    query_vector = embedding_response.data[0].embedding

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=num_results,
    )

    # Format results for the agent
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "chunk_id": i,
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "doc_type": results["metadatas"][0][i].get("doc_type", "unknown"),
            "headline": results["metadatas"][0][i].get("headline", ""),
            "text": results["documents"][0][i],
        })

    return json.dumps({
        "query": query,
        "num_results": len(chunks),
        "chunks": chunks,
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Tool 2 — Keyword Search (exact string matching)
# ═══════════════════════════════════════════════════════════════════

@function_tool
def keyword_search(keyword: str) -> str:
    """
    Search all knowledge base files for an exact keyword or phrase.
    Scans every .md file and returns matching paragraphs.

    Use this when vector search misses something, or when you need
    to find a specific name, number, or exact term.

    Args:
        keyword: The exact word or phrase to search for (case-insensitive)
    """
    keyword_lower = keyword.lower()
    matches = []

    kb_path = Path(KNOWLEDGE_BASE_PATH)
    for md_file in kb_path.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        if keyword_lower in content.lower():
            # Find the paragraph(s) containing the keyword
            paragraphs = content.split("\n\n")
            for para in paragraphs:
                if keyword_lower in para.lower():
                    matches.append({
                        "source": md_file.name,
                        "doc_type": md_file.parent.name,
                        "matching_text": para.strip()[:800],
                    })

    return json.dumps({
        "keyword": keyword,
        "num_matches": len(matches),
        "matches": matches[:20],  # limit to 20 matches
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Tool 3 — List Sources (see what's available)
# ═══════════════════════════════════════════════════════════════════

@function_tool
def list_sources(doc_type: str = "") -> str:
    """
    List the documents available in the knowledge base.
    Optionally filter by document type (employees, products, contracts, company).

    Use this to understand what information is available before searching.

    Args:
        doc_type: Optional filter. Leave empty to see all types.
                  Examples: 'employees', 'products', 'contracts', 'company'
    """
    kb_path = Path(KNOWLEDGE_BASE_PATH)
    sources = {}

    for md_file in kb_path.rglob("*.md"):
        dtype = md_file.parent.name
        if doc_type and dtype != doc_type:
            continue
        if dtype not in sources:
            sources[dtype] = []
        sources[dtype].append(md_file.name)

    # Sort each list
    for dtype in sources:
        sources[dtype].sort()

    summary = {
        "total_files": sum(len(v) for v in sources.values()),
        "document_types": {k: len(v) for k, v in sources.items()},
        "files": sources,
    }

    return json.dumps(summary, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Tool 4 — Rerank Chunks (LLM reorders by relevance)
# ═══════════════════════════════════════════════════════════════════

class RankOrder(BaseModel):
    """The order of chunk IDs from most relevant to least relevant."""
    ordered_ids: list[int] = Field(
        description="List of chunk_id values, most relevant first"
    )


@function_tool
def rerank_chunks(question: str, chunks_json: str, top_k: int = 8) -> str:
    """
    Rerank a set of retrieved chunks by relevance to the question.
    An LLM reads every chunk and reorders them, putting the most useful ones first.
    Then only the top_k most relevant are kept.

    Use this AFTER vector_search when you have many chunks and want to
    focus on the most relevant ones.

    Args:
        question: The user's original question
        chunks_json: The JSON string from a previous vector_search result
        top_k: How many chunks to keep after reranking (default 8)
    """
    try:
        data = json.loads(chunks_json)
        chunks = data.get("chunks", [])
    except (json.JSONDecodeError, KeyError):
        return json.dumps({"error": "Could not parse chunks_json. Pass the exact output from vector_search."})

    if len(chunks) <= top_k:
        return chunks_json  # No need to rerank if we have fewer than top_k

    # Build the reranking prompt
    system_prompt = (
        "You are a document relevance ranker. You will be given a question "
        "and a list of text chunks. Rank the chunks by relevance to the question. "
        "Most relevant first. Reply ONLY with the ordered list of chunk_id numbers."
    )

    chunk_descriptions = []
    for c in chunks:
        chunk_descriptions.append(f"chunk_id={c['chunk_id']}:\n{c['text'][:500]}")

    user_prompt = (
        f"Question: {question}\n\n"
        f"Chunks:\n" + "\n---\n".join(chunk_descriptions)
    )

    try:
        response = client.beta.chat.completions.parse(
            model=UTILITY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=RankOrder,
        )
        order = response.choices[0].message.parsed.ordered_ids

        # Reorder chunks and keep only top_k
        id_to_chunk = {c["chunk_id"]: c for c in chunks}
        reranked = []
        for cid in order:
            if cid in id_to_chunk and len(reranked) < top_k:
                reranked.append(id_to_chunk[cid])

        return json.dumps({
            "question": question,
            "num_results": len(reranked),
            "reranked": True,
            "chunks": reranked,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Reranking failed: {e}", "chunks": chunks[:top_k]})


# ═══════════════════════════════════════════════════════════════════
# Tool 5 — Self-Evaluate (score the answer, suggest improvements)
# ═══════════════════════════════════════════════════════════════════

class EvalScores(BaseModel):
    """Evaluation scores for an answer."""
    accuracy: int = Field(description="Score 1-5: Is the answer factually correct?")
    relevance: int = Field(description="Score 1-5: Does it address the question?")
    completeness: int = Field(description="Score 1-5: Does it cover all key points?")
    feedback: str = Field(description="Specific suggestions for improvement, or 'Excellent' if all scores are 5")


@function_tool
def self_evaluate(question: str, answer: str, context: str = "") -> str:
    """
    Evaluate an answer for quality. Returns scores and improvement suggestions.

    Use this AFTER you have drafted an answer to check if it's good enough.
    If any score is below 4, you should search for more context and try again.

    Args:
        question: The user's original question
        answer: Your proposed answer to evaluate
        context: The retrieved context you used (optional, helps evaluation)
    """
    system_prompt = """You are an answer quality evaluator. 
Score the answer on three dimensions (1-5 each):
- accuracy: Is it factually correct based on the context?
- relevance: Does it directly address the question asked?
- completeness: Does it cover all important aspects?

If any dimension scores below 4, provide SPECIFIC suggestions for what's missing 
or wrong and what additional information to search for.
If all scores are 4+, say 'Excellent'."""

    user_prompt = f"""Question: {question}

Answer to evaluate: {answer}

Context used: {context[:3000] if context else 'Not provided'}"""

    try:
        response = client.beta.chat.completions.parse(
            model=UTILITY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=EvalScores,
        )
        scores = response.choices[0].message.parsed

        passed = all(
            getattr(scores, dim) >= MIN_ACCEPTABLE_SCORE
            for dim in ["accuracy", "relevance", "completeness"]
        )

        return json.dumps({
            "accuracy": scores.accuracy,
            "relevance": scores.relevance,
            "completeness": scores.completeness,
            "passed": passed,
            "feedback": scores.feedback,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Evaluation failed: {e}", "passed": True})