# Recursive Chunking & RAG Laboratory

`recursive_chunking_rag_lab.ipynb` is a self-explaining notebook that lets you *watch* recursive text chunking
happen and then uses the chunks in a complete local RAG pipeline.

**Pipeline:** 1210-character document → statistics → custom recursive splitter (fully traced) → recursion tree →
5 chunk sizes → local MiniLM embeddings → local Qdrant → manual cosine vs Qdrant search → retrieval evaluation
(Recall@1/3/5, MRR) → Groq generation → LangChain comparison.

## Run it
```bash
pip install -r requirements.txt
export GROQ_API_KEY="..."        # optional; without it the generation step is skipped gracefully
jupyter lab recursive_chunking_rag_lab.ipynb   # Run All
```
* The first run downloads `sentence-transformers/all-MiniLM-L6-v2` (~90 MB); later runs use the local cache.
* In Google Colab, add `GROQ_API_KEY` under *Secrets* (the notebook reads it automatically).
* Optional: `GROQ_MODEL` overrides the default model (`llama-3.3-70b-versatile`).
* The first code cell installs any missing package, so `pip install` is optional.

## What to look at
| Section | Why it matters |
|---|---|
| §9 Detailed Recursion Trace | every recursive call, separator decision, keep/recurse verdict, merge step |
| §10 Visual Recursion Tree | generated from the trace (text + coloured HTML) |
| §11 `trace_chunk(...)` | final chunk → merge → pieces → recursive calls → original characters |
| §14–15, §23 | 5 chunk sizes, evidence integrity matrix, retrieval metrics, selection heuristic, overlap sweep |
| §26 | custom splitter vs LangChain `RecursiveCharacterTextSplitter` |
| last cell | programmatic requirements checklist |

## Notes
* All results in the notebook are computed at run time; no numbers are hard-coded.
* The selection score weights (70% retrieval / 30% structure) are editable constants in §15.
* Developer switches (leave unset normally): `RAGLAB_OFFLINE_TEST=1`, `RAGLAB_ALLOW_FALLBACK_EMBEDDER=1`
  let the notebook run in a sandbox without model downloads using a lexical stand-in embedder (clearly labelled, **not** MiniLM).
