"""
ingest.py — Build the vector database from your knowledge base.

This script does three things:
  1. Uses LangChain to load all .md files from the knowledge base
  2. Calls an LLM to semantically chunk each document (with summaries)
  3. Stores the chunks and their vectors in ChromaDB

Run this once (or whenever your knowledge base changes):
    python ingest.py
"""

import json
from pathlib import Path
from multiprocessing import Pool
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

# LangChain is used here for convenient document loading
from langchain_community.document_loaders import DirectoryLoader, TextLoader

import chromadb

from config import (
    KNOWLEDGE_BASE_PATH,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    UTILITY_MODEL,
    EMBEDDING_MODEL,
    AVG_CHUNK_SIZE,
    INGEST_WORKERS,
)

client = OpenAI()


# ═══════════════════════════════════════════════════════════════════
# Pydantic models for structured outputs
# ═══════════════════════════════════════════════════════════════════

class Chunk(BaseModel):
    """One meaningful section of a document."""
    headline: str = Field(description="A brief heading for this chunk, a few words")
    summary: str = Field(description="A 2-3 sentence summary of the chunk content")
    original_text: str = Field(description="The original source text from the document")


class ChunkList(BaseModel):
    """A list of chunks extracted from a single document."""
    chunks: list[Chunk]


# ═══════════════════════════════════════════════════════════════════
# Step 1 — Load documents using LangChain
# ═══════════════════════════════════════════════════════════════════

def load_documents() -> list[dict]:
    """
    Uses LangChain's DirectoryLoader to recursively find and load
    all .md files from the knowledge base folder.

    Returns a list of dicts with 'content', 'source', and 'doc_type'.
    """
    print(f"Loading documents from: {KNOWLEDGE_BASE_PATH}")

    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_PATH),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    langchain_docs = loader.load()

    documents = []
    for doc in langchain_docs:
        source_path = Path(doc.metadata.get("source", ""))
        # The folder name tells us the document type (employees, products, etc.)
        doc_type = source_path.parent.name if source_path.parent.name != "" else "general"

        documents.append({
            "content": doc.page_content,
            "source": source_path.name,
            "doc_type": doc_type,
        })

    print(f"Loaded {len(documents)} documents")
    return documents


# ═══════════════════════════════════════════════════════════════════
# Step 2 — Semantic chunking with an LLM
# ═══════════════════════════════════════════════════════════════════

def _build_chunking_prompt(doc: dict) -> list[dict]:
    """Build the prompt that asks the LLM to split a document into chunks."""

    # Estimate how many chunks would be reasonable
    estimated_chunks = max(2, len(doc["content"]) // AVG_CHUNK_SIZE)

    user_msg = f"""You are given a document from a company knowledge base.
The document is of type: {doc['doc_type']}
The source file is: {doc['source']}

Split this document into approximately {estimated_chunks} meaningful, slightly overlapping chunks.
Each chunk should represent a coherent topic or section.

For each chunk, provide:
- headline: a brief heading (a few words)
- summary: a 2-3 sentence summary written so it will match well with search queries
- original_text: the original text from the document

Here is the document:

{doc['content']}

Respond with the chunks."""

    return [{"role": "user", "content": user_msg}]


def chunk_one_document(doc: dict) -> list[dict]:
    """
    Send one document to the LLM and get back semantic chunks.
    Uses structured outputs so the response is guaranteed to be valid JSON.
    """
    messages = _build_chunking_prompt(doc)

    try:
        response = client.beta.chat.completions.parse(
            model=UTILITY_MODEL,
            messages=messages,
            response_format=ChunkList,
        )
        chunk_list = response.choices[0].message.parsed

        # Convert each chunk into a dict ready for ChromaDB
        results = []
        for chunk in chunk_list.chunks:
            # Combine headline + summary + original text for embedding.
            # This gives the embedding model rich text to work with.
            combined_text = (
                f"## {chunk.headline}\n\n"
                f"{chunk.summary}\n\n"
                f"{chunk.original_text}"
            )
            results.append({
                "text": combined_text,
                "headline": chunk.headline,
                "source": doc["source"],
                "doc_type": doc["doc_type"],
            })
        return results

    except Exception as e:
        print(f"  Error chunking {doc['source']}: {e}")
        # Fallback: just use the raw document as a single chunk
        return [{
            "text": doc["content"],
            "headline": doc["source"],
            "source": doc["source"],
            "doc_type": doc["doc_type"],
        }]


def chunk_all_documents(documents: list[dict]) -> list[dict]:
    """
    Process all documents in parallel using multiprocessing.
    Each document is sent to the LLM for semantic chunking.
    """
    print(f"Chunking {len(documents)} documents with {INGEST_WORKERS} workers...")

    all_chunks = []
    with Pool(INGEST_WORKERS) as pool:
        for result_batch in tqdm(
            pool.imap(chunk_one_document, documents),
            total=len(documents),
            desc="Semantic chunking",
        ):
            all_chunks.extend(result_batch)

    print(f"Created {len(all_chunks)} chunks")
    return all_chunks


# ═══════════════════════════════════════════════════════════════════
# Step 3 — Embed and store in ChromaDB
# ═══════════════════════════════════════════════════════════════════

def store_in_chromadb(chunks: list[dict]):
    """
    Create embeddings for all chunks and store them in ChromaDB.
    """
    print(f"Creating embeddings and storing in ChromaDB at {CHROMA_DB_PATH}...")

    # Connect to ChromaDB (persistent, saved to disk)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # Delete the old collection if it exists, start fresh
    try:
        chroma_client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

    # Get the text from each chunk
    texts = [c["text"] for c in chunks]

    # Call OpenAI to create embeddings (in batches of 100)
    all_embeddings = []
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])

    # Prepare metadata and IDs
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": c["source"],
            "doc_type": c["doc_type"],
            "headline": c["headline"],
        }
        for c in chunks
    ]

    # Add everything to ChromaDB
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB")
    print("Done! The vector database is ready.")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Step 1: Load with LangChain
    documents = load_documents()

    # Step 2: Semantic chunking with LLM
    chunks = chunk_all_documents(documents)

    # Step 3: Embed and store
    store_in_chromadb(chunks)