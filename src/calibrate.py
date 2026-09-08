"""
Calibration: measure score distributions and suggest thresholds for BOTH
first-stage retrieval (cosine similarity) and second-stage cross-encoder reranking.
Run from src/.
"""
import json
import random

import numpy as np

import _disable_torchvision  # noqa: F401
from retriever import Retriever
from reranker import Reranker
import config

# --- 25 In-Scope Questions (LLM Agents in cs.CL) ---
IN_SCOPE = [
    "What are common architectures for LLM-based agents?",
    "How do LLM agents use tools or external APIs?",
    "What evaluation benchmarks are used for LLM agents?",
    "How do agents plan and decompose complex tasks?",
    "What memory mechanisms do LLM agents use?",
    "How does ReAct combine reasoning and acting?",
    "How do multi-agent systems coordinate and communicate?",
    "What are common failure modes of LLM agents?",
    "How do agents learn from feedback or self-reflection?",
    "How are LLM agents used for web navigation and browsing?",
    "How do agents handle long-horizon tasks and error recovery?",
    "What role does self-reflection play in Reflexion?",
    "How do LLM agents interact with bash and software environments in SWE-bench?",
    "What prompting strategies improve tool-use accuracy in agents?",
    "How do tree search and Monte Carlo tree search apply to LLM agents?",
    "What are the security vulnerabilities and prompt injection risks in tool-using agents?",
    "How is short-term vs long-term memory structured in LLM agents?",
    "How do autonomous coding agents generate and debug code iteratively?",
    "What mechanisms prevent infinite loops or hallucinations in autonomous agents?",
    "How do multi-agent debates improve factual reasoning?",
    "What role do environment observations play in LLM action spaces?",
    "How are embodied agents trained to navigate virtual environments?",
    "What role does reinforcement learning from human feedback play in agent policy training?",
    "How do agents handle context length constraints when executing long trajectories?",
    "What benchmarks evaluate web-browsing agents on realistic websites?",
]

# Near-miss: NLP/ML topics that are NOT about agents
NEAR_MISS = [
    "How does BERT pretraining work?",
    "What metrics are used to evaluate machine translation?",
    "How is named entity recognition done with transformers?",
    "What is the difference between stemming and lemmatization?",
    "How do diffusion models generate images?",
    "How does automatic speech recognition work with Whisper?",
    "What is Byte Pair Encoding in tokenizers?",
    "How does RoPE positional embedding work?",
    "What are the differences between encoder-only and decoder-only architectures?",
    "How does LoRA low-rank adaptation work for fine-tuning?",
]

# Off-topic: completely unrelated to computer science
OFF_TOPIC = [
    "How do I cook pasta carbonara?",
    "Who won the 2018 World Cup?",
    "What is the capital of Australia?",
    "How do I change a flat tire?",
    "What causes the seasons on Earth?",
    "How do I treat a sprained ankle?",
    "What are the rules of chess?",
    "How do airplanes stay in the air?",
    "What is the best way to brew espresso?",
    "How do plants perform photosynthesis?",
]


def sweep_threshold(pos, neg):
    """Return (balanced_accuracy, threshold, tpr, tnr) for the best cut-off.
    'Positive' means score >= threshold. Ties go to the lowest threshold."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    best = None
    for t in np.unique(np.concatenate([pos, neg])):
        tpr = float((pos >= t).mean())   # in-scope correctly accepted
        tnr = float((neg < t).mean())    # negatives correctly rejected
        bal = (tpr + tnr) / 2
        if best is None or bal > best[0] + 1e-9:
            best = (bal, float(t), tpr, tnr)
    return best


def collect(r, reranker, questions):
    rows = []
    for q in questions:
        # 1. First-stage retrieval (FAISS + BM25)
        candidates = r.retrieve(q, top_k=config.RERANK_TOP_N)
        top1_cosine = max((c["score"] for c in candidates), default=0.0)
        mean_cosine = float(np.mean([c["score"] for c in candidates])) if candidates else 0.0

        # 2. Second-stage reranking (Cross-Encoder)
        reranked = reranker.rerank(q, candidates, top_k=config.TOP_K)
        top1_rerank = reranked[0]["rerank_score"] if reranked else -99.0
        mean_rerank = float(np.mean([c["rerank_score"] for c in reranked])) if reranked else -99.0
        top_title = reranked[0]["title"] if reranked else (candidates[0]["title"] if candidates else "None")

        rows.append({
            "q": q,
            "top1_cosine": top1_cosine,
            "mean_cosine": mean_cosine,
            "top1_rerank": top1_rerank,
            "mean_rerank": mean_rerank,
            "title": top_title,
        })
    return rows


def show(name, rows):
    print(f"\n=== {name} ===")
    for x in rows:
        print(f"  cosine={x['top1_cosine']:.3f} | rerank={x['top1_rerank']:+.3f} | {x['q'][:45]:<45} -> {x['title'][:40]}")
    c_scores = np.array([x["top1_cosine"] for x in rows])
    r_scores = np.array([x["top1_rerank"] for x in rows])
    print(f"  Cosine top-1 min/median/max: {c_scores.min():.3f} / {np.median(c_scores):.3f} / {c_scores.max():.3f}")
    print(f"  Rerank top-1 min/median/max: {r_scores.min():+.3f} / {np.median(r_scores):+.3f} / {r_scores.max():+.3f}")


if __name__ == "__main__":
    print(f"Loading retriever and reranker for calibration...")
    print(f"Dataset sizes: {len(IN_SCOPE)} in-scope, {len(NEAR_MISS)} near-miss, {len(OFF_TOPIC)} off-topic.")

    r = Retriever()
    reranker = Reranker()

    ins = collect(r, reranker, IN_SCOPE)
    near = collect(r, reranker, NEAR_MISS)
    off = collect(r, reranker, OFF_TOPIC)

    show("IN-SCOPE (Relevant Agent Questions)", ins)
    show("NEAR-MISS (NLP/ML but Not Agents)", near)
    show("OFF-TOPIC (Non-CS Negatives)", off)

    # 1. Cosine Calibration (SIMILARITY_THRESHOLD for arXiv live fallback)
    c_pos = [x["top1_cosine"] for x in ins]
    c_neg = [x["top1_cosine"] for x in near + off]

    bal_c, t_cosine, tpr_c, tnr_c = sweep_threshold(c_pos, c_neg)
    floor_c = float(np.floor((np.percentile(c_pos, 5) - 0.02) * 100) / 100)
    suggested_sim_threshold = max(t_cosine, floor_c)

    # 2. Reranker Calibration (MIN_RERANK_SCORE for gating LLM context)
    r_pos = [x["top1_rerank"] for x in ins]
    r_neg = [x["top1_rerank"] for x in near + off]

    bal_r, t_rerank, tpr_r, tnr_r = sweep_threshold(r_pos, r_neg)
    suggested_rerank_threshold = round(t_rerank, 2)

    print("\n================ SUGGESTED config.py VALUES ================")
    print(f"SIMILARITY_THRESHOLD = {suggested_sim_threshold:.2f}   # Cosine fallback gate: in-scope kept {tpr_c:.0%}, negatives rejected {tnr_c:.0%}")
    print(f"MIN_RERANK_SCORE     = {suggested_rerank_threshold:+.2f}   # Cross-encoder gate: in-scope kept {tpr_r:.0%}, negatives rejected {tnr_r:.0%}")
    print("============================================================")