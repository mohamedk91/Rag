# LLM Agents Research Assistant

A local retrieval-augmented generation (RAG) project for asking questions about LLM-agent research papers from arXiv's `cs.CL` category. It builds a searchable paper collection, retrieves relevant passages with dense and keyword search, reranks them, and generates answers with source citations. A Gradio chat interface and a command-line interface use the same pipeline.

> **Hardware note:** The default configuration assumes two CUDA GPUs: the Qwen3-8B generator is placed on `cuda:0`, while the embedding and reranking models use `cuda:1`. The generator loads in 4-bit NF4 quantization. This is not a CPU-first setup; change the device settings and model choices in `src/config.py` to suit your hardware. `bitsandbytes` and CUDA support depend on the operating system and PyTorch installation.

## How it works

1. Fetches paper metadata and PDFs from arXiv (`cs.CL` and the configured `LLM agents` search phrase).
2. Converts PDFs to Markdown with `pymupdf4llm`.
3. Splits paper text into section-aware chunks and filters likely references, short text, table dumps, and symbol-heavy content.
4. Embeds chunks with `Qwen/Qwen3-Embedding-0.6B` and stores normalized vectors in a FAISS inner-product index.
5. Combines dense retrieval and BM25 results with reciprocal rank fusion, then reranks candidates using `BAAI/bge-reranker-v2-m3`.
6. Uses `Qwen/Qwen3-8B` to generate answers grounded in retrieved passages. The pipeline refuses when no passages meet its reranker score threshold and validates generated arXiv IDs against retrieved sources.

When enabled, live fallback searches arXiv if the best local dense similarity is below the configured threshold. It filters candidates to those mentioning agents, scores title and abstract relevance, then downloads, parses, chunks, and adds selected papers to the local index. This requires network access at runtime and writes to `data/`.

## Requirements

- Python 3.10 or newer is recommended.
- Dependencies are listed in [`requirements.txt`](requirements.txt).
- A CUDA-capable setup with enough GPU memory for the selected models is expected with the defaults. Two GPUs are assumed by the default device mapping.
- Initial corpus creation and live fallback need access to the arXiv API and paper PDFs. Model loading may also download model files from Hugging Face on first use.

Install from the project root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA/OS before running the project if the default package resolution does not provide the needed GPU support. In particular, `bitsandbytes` platform support varies.

### Two-GPU setup (Windows PowerShell)

The defaults in `src/config.py` map the generator to logical GPU 0 and both embedding and reranking to logical GPU 1. Keep both GPUs visible; the project does not automatically choose devices or split one model across the cards.

```powershell
nvidia-smi
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch build matching your driver/platform first.
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
$env:CUDA_VISIBLE_DEVICES = "0,1"
python src/1_fetch_papers.py
python src/2_parse_pdfs.py
python src/3_chunk.py
python src/4_embed_index.py
python src/app.py
```

The first run downloads paper and model files. Check that PyTorch reports at least two CUDA GPUs before running the embedding or app steps. `CUDA_VISIBLE_DEVICES` controls which physical cards map to logical `cuda:0` and `cuda:1`. Native Windows support for 4-bit `bitsandbytes` can vary; if it fails, use a supported Linux/WSL2 GPU environment or adjust the quantization setup.

## Build the local corpus and index

Run these commands from the project root, in order:

```bash
python src/1_fetch_papers.py
python src/2_parse_pdfs.py
python src/3_chunk.py
python src/4_embed_index.py
```

The stages can be rerun; fetching and parsing skip files already present, while chunking and index building regenerate their outputs. The default fetch cap and search scope are configured in `src/config.py` (`MAX_PAPERS`, `ARXIV_CATEGORY`, and `ARXIV_SEARCH_QUERY`).

Generated files are stored under `data/`, which is ignored by Git:

| Path | Contents |
| --- | --- |
| `data/papers_pdf/` | Downloaded PDFs |
| `data/papers_md/` | Extracted Markdown |
| `data/chunks/chunks.jsonl` | Section-aware chunk records |
| `data/faiss.index` | Dense vector index |
| `data/chunks_store.jsonl` | Chunk records aligned with FAISS IDs |
| `data/papers_metadata.json` | Paper metadata |

The retrieval pipeline expects a built index and chunk store. Running `4_embed_index.py` requires `chunks.jsonl` produced by the earlier stages.

## Ask questions

Interactive command-line session:

```bash
python src/rag_pipeline.py
```

Type a question and enter `quit` or `exit` to close it.

Launch the Gradio chat app:

```bash
python src/app.py
```

The app loads the models at startup and prints its local URL. Requests are queued and processed one at a time. To enable a public Gradio share link, set `RAG_SHARE=1`. If you do this, configure both `RAG_USER` and `RAG_PASSWORD` to require login. Public sharing also disables live fallback by default.

Runtime options (read from environment):

| Variable | Effect |
| --- | --- |
| `RAG_SHARE` | Enables Gradio's public share link when set to `1`, `true`, or `yes`. Default: off. |
| `RAG_ALLOW_FALLBACK` | Enables or disables live arXiv fallback. Default: enabled for private use and disabled when `RAG_SHARE` is enabled. |
| `RAG_USER` / `RAG_PASSWORD` | When both are set, enables Gradio basic authentication. |

Other model, device, retrieval, and fallback settings are in `src/config.py`.

## MCP server integration

**MCP is not implemented in this repository yet.** The current entry points are the Gradio app and CLI; there is no MCP SDK dependency, server module, transport, or MCP tool schema. To expose this RAG system to MCP clients, add a separate server adapter around `RagPipeline` rather than treating the Gradio app as an MCP endpoint.

A minimal MCP adapter should:

- Create and reuse one `RagPipeline` instance for the server process so models and the index are not reloaded for every tool call.
- Expose a typed `ask_research` tool accepting a question and, if needed, bounded conversation history. Return structured answer, refusal status, and source metadata (title, arXiv ID, section, and scores) rather than only formatted Markdown.
- Optionally expose a read-only `search_passages` tool for callers that need retrieved evidence without generation. Validate input lengths and cap returned passages.
- Use stdio transport for a local MCP client, or an authenticated HTTP MCP transport for a remotely hosted service. Remote deployment needs TLS, identity/authentication, request limits, and deployment-specific network controls.
- Keep corpus ingestion and index mutation out of the default public tool surface. If an administrative ingestion tool is added, give it separate authorization and audit logging.
- Disable live fallback by default for shared MCP deployments, or place it behind explicit policy, timeouts, rate limits, and resource caps. Fallback downloads papers and modifies the local index.

The adapter would call the existing synchronous methods (`RagPipeline.ask` or retrieval methods) and can initially return a complete answer rather than streaming. MCP clients would need to be configured to launch/connect to the new server entry point; the exact command depends on the chosen SDK and transport. None of these MCP changes are present yet.

## Production readiness gaps

This project is a research/demo implementation, not yet a production service. Before serving real users, address at least the following:

1. **Deployment and hardware:** Package a reproducible runtime with an explicitly supported OS, CUDA/PyTorch combination, model cache strategy, GPU sizing, startup/readiness checks, graceful shutdown, and a deployment topology. Current defaults assume two local CUDA devices and load all models into one process.
2. **Identity and access control:** Add proper authentication and authorization for the chosen deployment. Gradio's optional username/password is only a basic demo gate; public share links should not be treated as a production security boundary. For a remote MCP transport, enforce client identity and authorization at the server boundary.
3. **Concurrency and persistence:** The locks protect operations within one process. The FAISS index and chunk files are local disk state; multiple worker processes or replicas can race, diverge, or overwrite updates. Use a single-writer ingestion path or a shared vector/search service with transactional versioned snapshots and backups before horizontal scaling.
4. **Ingestion lifecycle:** Separate scheduled/admin corpus updates from user query handling. Add durable job status, retries, validation, deduplication, rollback, and a way to publish a new corpus version atomically. Runtime fallback currently performs network calls and disk/index writes during a user request.
5. **Reliability and operations:** Replace ad hoc `print` diagnostics with structured logs, request IDs, metrics, tracing, health/readiness endpoints, timeout budgets, bounded queues, cancellation, and alerting. Define expected latency, availability, capacity, and recovery targets.
6. **Input and abuse controls:** Add per-user rate and token limits, payload limits, request timeouts, and controls against resource exhaustion. Review the whole path for prompt injection and unsafe content from downloaded papers; context instructions reduce risk but are not a complete security control.
7. **Model and dependency management:** `requirements.txt` uses lower bounds rather than a tested lockfile. Pin and scan dependencies, pin model revisions, record model/tokenizer/config versions with every index, review licenses and provenance, and support controlled updates and rollback.
8. **Quality gates:** Expand the labeled evaluation set and run regression tests on retrieval, groundedness, citation correctness, refusal behavior, latency, and cost for every model, prompt, corpus, or threshold change. The current tests are mostly unit-level and may mock unavailable heavy dependencies; they do not demonstrate production quality.
9. **Data governance:** Define retention, deletion, backup, access, and licensing policies for downloaded PDFs, extracted text, logs, and user queries. Provide a process to remove or refresh papers and rebuild the index consistently.

The items above are work still required; adding an MCP wrapper alone would make the pipeline callable by MCP clients but would not make it production-ready.

## Evaluation

The checked-in `eval/questions.json` contains labeled questions. Run evaluation commands from the project root:

```bash
# Retrieval and reranker metrics; saves eval/results/<tag>.json
python src/eval.py run --tag full_pipeline

# End-to-end answer and citation audit; saves a report under eval/answers/
python src/eval.py answers --n 30 --tag answer_audit

# Compare two previously saved retrieval runs
python src/eval.py compare baseline full_pipeline

# Optional: create a skeleton question set from indexed metadata
python src/eval.py sample --n 40
```

The sample command creates blank answerable questions that must be written before evaluation; it will not overwrite an existing `eval/questions.json` unless `--force` is supplied. `src/calibrate.py` can be run with `python src/calibrate.py` to inspect score distributions and help tune thresholds in `src/config.py`. Evaluation results and answer audits are generated artifacts.

## Tests

From the project root:

```bash
python -m pytest tests/
```

Some pipeline tests mock heavyweight dependencies when unavailable; passing the unit tests does not verify GPU model loading, arXiv access, or answer quality on a built corpus.

## Project layout

```text
src/
  1_fetch_papers.py   Fetch metadata and PDFs
  2_parse_pdfs.py     Convert PDFs to Markdown
  3_chunk.py          Create filtered section-aware chunks
  4_embed_index.py    Build FAISS embeddings and chunk store
  rag_pipeline.py     Retrieval, reranking, fallback, and answer generation
  app.py              Gradio chat interface
  eval.py             Retrieval and answer evaluation commands
  calibrate.py        Threshold calibration helper
  config.py           Paths, models, devices, and pipeline settings
tests/                Unit tests
eval/questions.json   Labeled evaluation questions
```


## Known limitations

- Answers depend on the downloaded papers and retrieved passages; citations identify retrieved sources and do not guarantee every claim is correct.
- The default model/device choices require substantial GPU resources and may need adjustment.
- Live fallback depends on arXiv availability and can modify the local paper corpus and index.
- Similarity and reranker thresholds in `src/config.py` are configuration values that may need calibration for a changed corpus or model setup.
- The project currently has no MCP server, remote API, production authentication system, or multi-replica index backend.
