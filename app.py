import asyncio
import gradio as gr
from agents import Runner

from agent import rag_agent


async def _stream_response(message: str, history: list) -> str:
    messages = []

    if history:
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    messages.append({"role": "user", "content": message})

    result = await Runner.run(rag_agent, input=messages)

    return result.final_output


def chat(message: str, history: list) -> str:
    return asyncio.run(_stream_response(message, history))


with gr.Blocks() as demo:

    gr.Markdown("""
    # 🤖 InsurElm Agentic RAG — Knowledge Worker
    
    This is an **agentic** RAG system. Instead of a fixed search-then-answer pipeline, 
    an AI agent **decides its own retrieval strategy** — choosing which tools to use, 
    running multiple searches, reranking results, and self-evaluating its answers.
    
    **Tools available to the agent:**
    🔍 Vector Search · 📝 Keyword Search · 📂 Document Listing · 🔄 Reranking · ✅ Self-Evaluation
    
    Try asking about employees, products, contracts, or company policies!
    """)

    chatbot = gr.ChatInterface(
        fn=chat,
        examples=[
            "Who is Avery Lancaster?",
            "Who won the IoT award?",
            "Who went to Manchester University?",
            "What products does InsurElm offer?",
            "How many employees does InsurElm have?",
            "Tell me about the contract with Omega Insurance Group",
        ],
    )


if __name__ == "__main__":
    demo.launch()