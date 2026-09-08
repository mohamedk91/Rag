"""
Gradio demo with streaming. Requests are processed one at a time (single GPU,
single index). Answers stream token-by-token as they are generated.

Environment variables: RAG_SHARE and RAG_ALLOW_FALLBACK are read in config.py;
RAG_USER and RAG_PASSWORD (the optional login) are read here in app.py.
"""
import os
import traceback

import gradio as gr

import config
from rag_pipeline import RagPipeline

print("Starting up - loading models (this can take a minute)...")
pipeline = RagPipeline()


def _format_result(result):
    """Convert a pipeline result dict to a markdown response string."""
    fallback_note = (
        "\n*(No good match locally — fetched fresh papers from arXiv for this question.)*\n"
        if result.get("used_live_fallback")
        else ""
    )

    if not result["sources"] or result.get("refused") or pipeline._is_no_info(result.get("answer", "")):
        return f"{result['answer']}\n{fallback_note}"

    sources_md = "\n".join(
        f"- **{s['title']}** (arXiv:{s['arxiv_id']}, section: {s['section']}, score: {s['score']:.3f})"
        for s in result["sources"]
    )
    return f"{result['answer']}\n{fallback_note}\n---\n**Retrieved passages** (the answer may use only some of these):\n{sources_md}"


def respond(question, history):
    try:
        for result in pipeline.ask_stream(question, history=history):
            yield _format_result(result)
    except Exception:
        traceback.print_exc()                      # full error stays in the server log
        yield "Something went wrong while answering. Please try again."


demo = gr.ChatInterface(
    fn=respond,
    title="LLM Agents Research Assistant (RAG over arXiv cs.CL)",
    description=(
        "Ask questions about LLM agent research papers indexed from arXiv (cs.CL). "
        "Answers are grounded in retrieved paper excerpts and cite their sources. "
        "Answers stream in real-time as they are generated."
    ),
    examples=[
        "What are common architectures for LLM-based agents?",
        "How do LLM agents use tools or external APIs?",
        "What evaluation benchmarks are used for LLM agents?",
    ],
)

# One request at a time; up to 10 people can wait in line, the rest are turned away.
demo.queue(default_concurrency_limit=1, max_size=10)


def _auth():
    user, password = os.environ.get("RAG_USER"), os.environ.get("RAG_PASSWORD")
    return (user, password) if user and password else None


if __name__ == "__main__":
    auth = _auth()
    if config.SHARE_PUBLIC and auth is None:
        print("WARNING: public link with no login. Set RAG_USER and RAG_PASSWORD to require one.")
    print(f"Public link: {config.SHARE_PUBLIC} | live fallback: {config.ALLOW_LIVE_FALLBACK} | login: {auth is not None}")
    demo.launch(share=config.SHARE_PUBLIC, auth=auth)