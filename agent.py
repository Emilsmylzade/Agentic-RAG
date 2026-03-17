import asyncio
from agents import Agent, Runner

from tools import (
    vector_search,
    keyword_search,
    list_sources,
    rerank_chunks,
    self_evaluate,
)
from config import AGENT_MODEL, MAX_EVAL_RETRIES


SYSTEM_PROMPT = f"""You are an expert knowledge worker for InsurElm, an insurance technology company.
You answer questions accurately using the company's internal knowledge base.

## Your Retrieval Strategy

You have 5 tools. Use them intelligently:

1. **Start with vector_search** — run a semantic search with the user's question.
   If the question is a follow-up, reformulate it into a standalone question first.

2. **Try keyword_search if needed** — if vector search didn't find a specific name,
   number, or term, use keyword_search with the exact term.

3. **Consider query expansion** — if your first search was too narrow, try another
   vector_search with a DIFFERENT phrasing of the question. For example, if the user
   asks "Who went to Manchester University?", also try searching for "University of Manchester".

4. **Rerank when you have many chunks** — if you retrieved lots of chunks, use
   rerank_chunks to sort them by relevance and keep only the best ones.

5. **Self-evaluate before finishing** — after drafting your answer, use self_evaluate
   to check its quality. If any score is below 4, search for more information and try again.
   You can retry up to {MAX_EVAL_RETRIES} times.

## Answer Guidelines

- Be accurate, relevant, and complete
- If the context doesn't contain the answer, say so honestly
- Use markdown formatting for clarity (tables, bold, etc.)
- Include specific details like names, dates, numbers when available
- Cite which document the information came from when possible

## Important Rules

- ALWAYS search before answering — never guess from your training data
- If a question spans multiple topics, do multiple searches
- The knowledge base contains: employees, products, contracts, and company info
- Today's date is available if needed for context
"""

rag_agent = Agent(
    name="InsurElm Knowledge Worker",
    instructions=SYSTEM_PROMPT,
    model=AGENT_MODEL,
    tools=[
        vector_search,
        keyword_search,
        list_sources,
        rerank_chunks,
        self_evaluate,
    ],
)


async def ask_agent(question: str, history: list[dict] = None) -> str:
    messages = []

    if history:
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    messages.append({"role": "user", "content": question})

    result = await Runner.run(rag_agent, input=messages)

    return result.final_output


def ask_agent_sync(question: str, history: list[dict] = None) -> str:
    return asyncio.run(ask_agent(question, history))


if __name__ == "__main__":
    print("Testing the Agentic RAG agent...")
    print("=" * 60)

    test_questions = [
        "Who is Avery Lancaster?",
        "Who won the IoT award?",
        "Who went to Manchester University?",
        "How many employees does InsurElm have?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        print("-" * 40)
        answer = ask_agent_sync(q)
        print(f"A: {answer}")
        print("=" * 60)