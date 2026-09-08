"""
Evaluation for the RAG project. Run from src/.

Subcommands:

  python eval.py sample --n 40
      Write a SKELETON question file with random papers and empty questions.

  python eval.py run --tag full_pipeline
      Evaluate first-stage hybrid retrieval (FAISS+BM25) AND second-stage
      cross-encoder reranking. Measures MRR, Hit@k, and llm_sees_it.

  python eval.py answers --n 20 --tag answer_audit
      Generate end-to-end answers for evaluated questions and run an answer-level
      faithfulness and citation audit (verifying cited arXiv IDs, refusal accuracy,
      and non-hallucination).

  python eval.py compare baseline full_pipeline
      Side-by-side metrics comparison between two saved evaluation runs.
"""
import argparse
import json
import os
import random
import re
import statistics
import time

import config

EVAL_DIR = os.path.join(config.PROJECT_ROOT, "eval")
QUESTIONS_PATH = os.path.join(EVAL_DIR, "questions.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
ANSWERS_DIR = os.path.join(EVAL_DIR, "answers")

EVAL_DEPTH = 30
PAPER_KS = (1, 3, 5)

DEFAULT_NEGATIVES = [
    ("n01", "near_miss", "How does BERT pretraining work?"),
    ("n02", "near_miss", "What metrics are used to evaluate machine translation?"),
    ("n03", "near_miss", "How is named entity recognition done with transformers?"),
    ("n04", "near_miss", "What is the difference between stemming and lemmatization?"),
    ("n05", "off_topic", "How do I cook pasta carbonara?"),
    ("n06", "off_topic", "Who won the 2018 World Cup?"),
    ("n07", "off_topic", "What is the capital of Australia?"),
    ("n08", "off_topic", "How do I change a flat tire?"),
]


# ---------------------------------------------------------------------------
# Metric logic
# ---------------------------------------------------------------------------

def paper_ranking(chunks):
    """Chunk list (best first) -> paper ids in order of FIRST appearance."""
    ranked = []
    for c in chunks:
        if c["arxiv_id"] not in ranked:
            ranked.append(c["arxiv_id"])
    return ranked


def score_question(chunks, relevant_ids, top_k_chunks):
    """Metrics for ONE answerable question. `chunks` is ordered best-first."""
    relevant = set(relevant_ids)
    ranking = paper_ranking(chunks)

    first_rank = next((i + 1 for i, pid in enumerate(ranking) if pid in relevant), None)
    metrics = {
        "first_rank": first_rank,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "recall@5": len(relevant & set(ranking[:5])) / len(relevant) if relevant else 0.0,
        "llm_sees_it": any(c["arxiv_id"] in relevant for c in chunks[:top_k_chunks]),
        "top1_score": chunks[0]["score"] if chunks else 0.0,
    }
    if chunks and "rerank_score" in chunks[0]:
        metrics["top1_rerank_score"] = chunks[0]["rerank_score"]

    for k in PAPER_KS:
        metrics[f"hit@{k}"] = bool(first_rank and first_rank <= k)
    return metrics


def aggregate(per_question):
    """Average per-question metrics for answerable questions."""
    ans = [q for q in per_question if q["type"] == "answerable"]
    out = {"n_answerable": len(ans)}
    if ans:
        for key in ("mrr", "recall@5", "hit@1", "hit@3", "hit@5", "llm_sees_it"):
            out[key] = statistics.mean(float(q["metrics"][key]) for q in ans)
        out["top1_score_median_answerable"] = statistics.median(q["metrics"]["top1_score"] for q in ans)
        if "top1_rerank_score" in ans[0]["metrics"]:
            out["top1_rerank_median_answerable"] = statistics.median(q["metrics"]["top1_rerank_score"] for q in ans)
    for t in ("near_miss", "off_topic"):
        scores = [q["metrics"]["top1_score"] for q in per_question if q["type"] == t]
        if scores:
            out[f"top1_score_median_{t}"] = statistics.median(scores)
            out[f"top1_score_max_{t}"] = max(scores)
        if ans and "top1_rerank_score" in per_question[0]["metrics"]:
            r_scores = [q["metrics"]["top1_rerank_score"] for q in per_question if q["type"] == t and "top1_rerank_score" in q["metrics"]]
            if r_scores:
                out[f"top1_rerank_median_{t}"] = statistics.median(r_scores)
                out[f"top1_rerank_max_{t}"] = max(r_scores)
    return out


# ---------------------------------------------------------------------------
# sample: build skeleton questions
# ---------------------------------------------------------------------------

def cmd_sample(args):
    if os.path.exists(QUESTIONS_PATH) and not args.force:
        raise SystemExit(f"{QUESTIONS_PATH} already exists. Use --force to overwrite it.")
    with open(config.METADATA_JSON, encoding="utf-8") as f:
        papers = json.load(f)

    rng = random.Random(args.seed)
    chosen = rng.sample(papers, min(args.n, len(papers)))

    entries = [{
        "id": f"q{i + 1:02d}",
        "question": "",
        "relevant_ids": [p["arxiv_id"]],
        "type": "answerable",
        "_title": p["title"],
        "_abstract": p["abstract"],
    } for i, p in enumerate(chosen)]
    entries += [{"id": i, "question": q, "relevant_ids": [], "type": t} for i, t, q in DEFAULT_NEGATIVES]

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(chosen)} skeleton questions + {len(DEFAULT_NEGATIVES)} negatives to {QUESTIONS_PATH}")


# ---------------------------------------------------------------------------
# run: evaluate retriever + reranker
# ---------------------------------------------------------------------------

def load_questions(known_ids):
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    questions, skipped = [], 0
    for q in raw:
        if not q.get("question", "").strip():
            skipped += 1
            continue
        if q["type"] == "answerable":
            if not q["relevant_ids"]:
                raise SystemExit(f"{q['id']}: answerable question has no relevant_ids")
            unknown = [i for i in q["relevant_ids"] if i not in known_ids]
            if unknown:
                raise SystemExit(f"{q['id']}: relevant_ids not in the index: {unknown}")
        questions.append(q)
    if skipped:
        print(f"(skipped {skipped} entries with an empty question)")
    return questions


def cmd_run(args):
    from retriever import Retriever
    from reranker import Reranker

    r = Retriever()
    reranker = Reranker()
    questions = load_questions(r.known_arxiv_ids)

    print(f"Evaluating {len(questions)} questions against {r.index.ntotal} chunks "
          f"from {len(r.known_arxiv_ids)} papers (Hybrid Retrieval + Cross-Encoder Reranking)...\n")

    per_question = []
    for q in questions:
        # 1. First stage: hybrid retrieval
        candidates = r.retrieve(q["question"], top_k=max(EVAL_DEPTH, config.RERANK_TOP_N))
        # 2. Second stage: cross-encoder reranking
        reranked = reranker.rerank(q["question"], candidates[:config.RERANK_TOP_N], top_k=config.TOP_K)

        if q["type"] == "answerable":
            metrics = score_question(reranked, q["relevant_ids"], config.TOP_K)
        else:
            metrics = {
                "top1_score": reranked[0]["score"] if reranked else 0.0,
                "top1_rerank_score": reranked[0]["rerank_score"] if reranked else -99.0,
            }

        per_question.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "relevant_ids": q["relevant_ids"], "metrics": metrics,
            "top_papers": [{"arxiv_id": c["arxiv_id"], "title": c["title"][:70], "section": c["section"][:40],
                            "score": round(c["score"], 3), "rerank_score": round(c.get("rerank_score", 0.0), 3)}
                           for c in reranked[:3]],
        })

    summary = aggregate(per_question)
    result = {
        "tag": args.tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "setup": {
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "reranker_model": config.RERANKER_MODEL_NAME,
            "top_k": config.TOP_K,
            "rerank_top_n": config.RERANK_TOP_N,
            "n_chunks": r.index.ntotal,
            "n_papers": len(r.known_arxiv_ids),
        },
        "summary": summary,
        "per_question": per_question,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"\nSaved: {out_path}")


def print_summary(s):
    print("=== SUMMARY (answerable questions) ===")
    print(f"  questions        : {s['n_answerable']}")
    print(f"  hit@1 / @3 / @5  : {s['hit@1']:.0%} / {s['hit@3']:.0%} / {s['hit@5']:.0%}")
    print(f"  recall@5         : {s['recall@5']:.0%}")
    print(f"  MRR              : {s['mrr']:.3f}")
    print(f"  llm_sees_it      : {s['llm_sees_it']:.0%}")
    print(f"  median top-1 cosine: {s['top1_score_median_answerable']:.3f}")
    if "top1_rerank_median_answerable" in s:
        print(f"  median top-1 rerank: {s['top1_rerank_median_answerable']:+.3f}")
    for t in ("near_miss", "off_topic"):
        if f"top1_rerank_median_{t}" in s:
            print(f"  top-1 rerank {t:<9}: median {s[f'top1_rerank_median_{t}']:+.3f}, max {s[f'top1_rerank_max_{t}']:+.3f}")


# ---------------------------------------------------------------------------
# answers: end-to-end answer quality & citation faithfulness evaluation
# ---------------------------------------------------------------------------

def check_citations(answer: str, provided_sources: list) -> dict:
    """
    Verify whether citations in the generated answer correspond to actual
    retrieved papers (preventing hallucinated citations).
    """
    valid_ids = {s["arxiv_id"].lower() for s in provided_sources}
    cited_ids = set(re.findall(r"\b(\d{4}\.\d{4,5})\b", answer))

    hallucinated = [cid for cid in cited_ids if cid.lower() not in valid_ids]
    valid_cited = [cid for cid in cited_ids if cid.lower() in valid_ids]

    return {
        "cited_ids": list(cited_ids),
        "valid_citations": valid_cited,
        "hallucinated_citations": hallucinated,
        "citation_fidelity": len(valid_cited) / len(cited_ids) if cited_ids else 1.0,
    }


def cmd_answers(args):
    from rag_pipeline import RagPipeline

    pipeline = RagPipeline()
    questions = load_questions(pipeline.retriever.known_arxiv_ids)
    if args.n:
        questions = questions[:args.n]

    print(f"\nEvaluating end-to-end answers for {len(questions)} questions...")
    results = []
    correct_refusals = 0
    total_negatives = 0
    total_citations = 0
    hallucinated_citations_count = 0

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']} ({q['type']}): {q['question'][:60]}...")
        out = pipeline.ask(q["question"])

        is_neg = q["type"] in ("near_miss", "off_topic")
        if is_neg:
            total_negatives += 1
            if out["refused"]:
                correct_refusals += 1

        cit_info = check_citations(out["answer"], out["sources"])
        total_citations += len(cit_info["cited_ids"])
        hallucinated_citations_count += len(cit_info["hallucinated_citations"])

        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "refused": out["refused"],
            "answer": out["answer"],
            "best_score": out["best_score"],
            "sources_count": len(out["sources"]),
            "citations": cit_info,
            "human_grade": None,  # Slot for manual review (1 to 5)
            "notes": "",
        })

    refusal_acc = (correct_refusals / total_negatives) if total_negatives else 1.0
    cit_precision = (
        (total_citations - hallucinated_citations_count) / total_citations
        if total_citations else 1.0
    )

    audit_summary = {
        "total_evaluated": len(questions),
        "total_negatives": total_negatives,
        "negative_refusal_accuracy": refusal_acc,
        "total_citations_found": total_citations,
        "hallucinated_citations": hallucinated_citations_count,
        "citation_precision": cit_precision,
    }

    os.makedirs(ANSWERS_DIR, exist_ok=True)
    out_file = os.path.join(ANSWERS_DIR, f"{args.tag}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": audit_summary, "results": results}, f, indent=2, ensure_ascii=False)

    print("\n=== ANSWER QUALITY AUDIT SUMMARY ===")
    print(f"  Questions evaluated          : {len(questions)}")
    print(f"  Negative refusal accuracy    : {refusal_acc:.0%} ({correct_refusals}/{total_negatives})")
    print(f"  Total citations checked      : {total_citations}")
    print(f"  Hallucinated citations count : {hallucinated_citations_count}")
    print(f"  Citation Precision           : {cit_precision:.0%}")
    print(f"\nDetailed report saved to: {out_file}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def cmd_compare(args):
    def load(tag):
        with open(os.path.join(RESULTS_DIR, f"{tag}.json"), encoding="utf-8") as f:
            return json.load(f)

    a, b = load(args.a), load(args.b)
    print(f"{'metric':<32}{args.a:>14}{args.b:>14}{'change':>10}")
    for key in ("hit@1", "hit@3", "hit@5", "recall@5", "mrr", "llm_sees_it"):
        va, vb = a["summary"][key], b["summary"][key]
        print(f"{key:<32}{va:>14.3f}{vb:>14.3f}{vb - va:>+10.3f}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="write a skeleton question file")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_sample)

    r = sub.add_parser("run", help="evaluate the current index + reranker")
    r.add_argument("--tag", required=True, help="name for this run, e.g. full_pipeline")
    r.set_defaults(fn=cmd_run)

    ans = sub.add_parser("answers", help="run end-to-end answer quality & citation fidelity audit")
    ans.add_argument("--n", type=int, default=30, help="number of questions to evaluate")
    ans.add_argument("--tag", default="audit", help="name for the answer audit report")
    ans.set_defaults(fn=cmd_answers)

    c = sub.add_parser("compare", help="compare two saved runs")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(fn=cmd_compare)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()