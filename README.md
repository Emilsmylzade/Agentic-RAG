# Agentic RAG

An agentic Retrieval-Augmented Generation (RAG) system built with the **OpenAI Agents SDK**, **ChromaDB**, and **Gradio**.

Unlike a fixed retrieve-then-answer pipeline, this project lets an AI agent decide **how to search**, **when to rerank**, **when to use keyword search**, and **when to self-evaluate its answer before responding**.

This project is based on an internal company-style knowledge base for **InsurElm**, containing documents about employees, products, contracts, and company information.

---

## Features

- **Agentic retrieval**
  - The agent chooses which tools to call and in what order
- **Semantic vector search**
  - Uses embeddings + ChromaDB for similarity-based retrieval
- **Keyword search**
  - Helps recover exact names, terms, or phrases that vector search may miss
- **Semantic chunking**
  - Documents are split into meaningful chunks using an LLM instead of fixed-size chunks
- **LLM-generated chunk summaries**
  - Each chunk is rewritten to become more searchable
- **Reranking**
  - Retrieved chunks can be reordered by relevance before answering
- **Self-evaluation**
  - The agent scores its own answer for accuracy, relevance, and completeness
- **Chat UI**
  - Simple Gradio interface for interacting with the system

---

## Why this project?

Traditional RAG pipelines often fail on buried details. A fact may exist in the source documents, but if it is hidden inside a long chunk, it may rank too low and never reach the model.

This project explores a more advanced approach:

1. **Create better chunks**
2. **Search in multiple ways**
3. **Rerank retrieved context**
4. **Let an agent decide the retrieval strategy**
5. **Evaluate answers before returning them**

That makes the system more flexible and better at handling harder questions than a rigid one-pass pipeline.

---

## Project structure

```bash
Agentic-RAG/
│
├── agent.py          # Main agent definition and orchestration
├── app.py            # Gradio chat interface
├── config.py         # Models, paths, and system settings
├── ingest.py         # Document loading, semantic chunking, embeddings, ChromaDB storage
├── tools.py          # Retrieval, keyword search, reranking, self-evaluation tools
├── README.md
│
└── knowledge-base/
    ├── company/
    ├── contracts/
    ├── employees/
    └── products/
