# AI Engineering Journey — Learning Progress

_Last updated: 18 November 2026_

## Purpose

This file is the handover reference for continuing the hands-on AI Engineering Journey in a **new ChatGPT conversation**. The local project and committed GitHub code are the source of truth if a detail here differs from the working implementation.

## Current status

- **Completed:** Lessons 1–11.
- **Lesson 12 — Advanced Document Processing & Chunking:** All eight instructional parts implemented; PDF ingestion and two live RAG questions demonstrated. Citation formatting and automated tests remain recommended improvements before treating the PDF RAG application as production-ready.
- **Next:** Lesson 13 — Advanced RAG (hybrid search, BM25, reranking, relevance filtering, and query rewriting).
- **Learning style:** Hands-on, step-by-step Python exercises, explanations, code review, descriptive module docstrings, minimal mathematics. Prefer attempting exercises before receiving full code unless explicitly requested.

## Repository and environment

- GitHub repository: https://github.com/learnaiengineer19/ai-engineering-journey
- Default branch: `main`
- Local working directory: `~/ai-engineering-journey`
- Virtual environment: `.venv`; activate with `source .venv/bin/activate`
- Run scripts from the repository root: `python <folder>/<script>.py`
- Development: macOS, VS Code, Python environment previously reported as 3.14.6
- Models served locally via Ollama:
  - `nomic-embed-text` — embedding model, 768 dimensions in this project
  - `qwen2:7b` — generative model
- Key Python packages: `ollama`, `chromadb`, `pymupdf`, `scikit-learn`, `pydantic`, `httpx` and built-in `unittest`.
- Git status, actual committed files, and GitHub push status should be checked locally; do not assume the latest changes have already been pushed.

## Lesson history

| Lesson | Topic | Outcome |
|---|---|---|
| 1 | Python variables and types | Fundamentals |
| 2 | Lists and dictionaries | LLM message-style structures |
| 3 | Functions, loops, JSON | Reusable code |
| 4 | Exceptions and input validation | Error handling |
| 5 | Modules and file I/O | Document processing basics |
| 6 | Classes, type hints and tests | Python `unittest` |
| 7 | Git and GitHub | Repository workflow |
| 8 | LLM fundamentals | Tokens, temperature, context windows |
| 9 | Ollama and structured output | `chat()`, JSON, Pydantic claim extraction |
| 10 | Embeddings and first RAG | JSON-persisted vectors, cosine similarity, cached search, `ask_rag()` and four passing tests |
| 11 | ChromaDB | Persistent collections, similarity search, metadata filters, add/update/upsert/delete, Chroma RAG, five passing tests |
| 12 | PDF document processing | PDF extraction, cleaning, sliding-window chunking, metadata, embedding ingestion, PDF-grounded RAG |

## Lesson 11 — working reference

- Database path used in lessons: `vector_databases/chroma_data`
- Collection for small insurance demo: `insurance_documents`, containing five documents.
- Separate practice collection: `insurance_documents_practice` (mutated during CRUD exercises).
- Important scripts may include `vector_databases/search_chroma.py`, `metadata_filtering.py`, `update_documents.py`, `delete_documents.py`, `chroma_rag.py`, and `test_chroma_rag.py`.
- Unit test command:

  ```bash
  python -m unittest discover -s vector_databases -p 'test_chroma_rag.py' -v
  ```

- **Confirmed result:** five tests passed (`test_valid_question`, `test_empty_query`, `test_invalid_query_type`, `test_no_documents`, `test_ollama_failure`).
- The demo RAG correctly abstained when retrieved documents did not establish engine replacement coverage. A separate answerable-query run exposed extra unsupported statements or missing citations, motivating stricter grounding instructions.

## Lesson 12 — PDF RAG

### Input PDF

- Training file: `document_processing/documents/insurance_policy.pdf`
- Fictional four-page motor insurance sample (not an actual policy).
- PDF sections include accident claims and evidence (page 2) and mechanical-breakdown exclusions (section 3.1, page 3).
- A copy of the lesson's PDF should be kept in the project if its redistribution is appropriate. Do not commit real customer or private insurance documents.

### Approximate file layout

```text
ai-engineering-journey/
├── llm_fundamentals/
├── vector_databases/
│   ├── chroma_rag.py
│   ├── test_chroma_rag.py
│   └── chroma_data/               # local generated DB; gitignored
├── document_processing/
│   ├── documents/
│   │   └── insurance_policy.pdf  # fictional lesson sample
│   ├── pdf_extractor.py
│   ├── text_cleaner.py
│   ├── text_chunker.py
│   ├── pdf_chunker.py
│   ├── ingest_pdf.py
│   └── pdf_rag.py
└── LEARNING_PROGRESS.md
```

This is a reference layout; confirm actual tracked files with `git status` and `git ls-files`.

### Processing pipeline

1. Extract per-page text with PyMuPDF `page.get_text('text', sort=True)`.
2. Normalize spaces and blank lines with `clean_text()`; preserve meaningful policy text.
3. Split each page into overlapping character chunks with `chunk_text()`.
4. Attach `chunk_id`, `source`, `page_number`, `chunk_index`, and chunk `text`.
5. Call Ollama `embed(model='nomic-embed-text', input=documents)` in a batch.
6. `upsert()` embeddings, texts, IDs and metadata into persistent Chroma collection `insurance_policy_chunks`.
7. At question time, embed the query, `collection.query()` for top-k chunks, construct source-aware context, and use Qwen2 to answer.

### Configuration and verified results

- Chunk size: **500 characters**; overlap: **100 characters**.
- Total generated PDF chunks: **16**; unique chunk IDs: **16**.
- Embedding dimension: **768**.
- Stored Chroma records: **16** in `insurance_policy_chunks`.
- Ingestion command:

  ```bash
  python document_processing/ingest_pdf.py
  ```

- PDF RAG command:

  ```bash
  python document_processing/pdf_rag.py
  ```

### Live question tests

**Accident claim documents:**

- Question: `What documents are required for an accident claim?`
- Retrieved three chunks from **PDF page 2**, with distances approximately **0.3791, 0.6218, 0.6466**.
- Qwen2 stated that the claim involves photographs of vehicle damage, completed claim form, incident date/location, and a police report under specified circumstances.
- Issue: Answer omitted expected `[Source N]` references; cite generation and validation should be improved.

**Engine breakdown:**

- Question: `Is mechanical engine breakdown covered?`
- Qwen2 response: Mechanical failure from wear and tear, breakdown, or lack of maintenance is excluded; damage directly caused by an insured collision may be considered, subject to policy terms.
- It generated `[Source 1:3.1]`, rather than the desired `[Source 1]` format.
- The source reference should be checked against the actual retrieved chunk and its metadata; do not assume all citations are correct simply because they look plausible.

### Known limitations to revisit

- Character-level chunking can split words or sentences; consider sentence/paragraph-aware splitting in Lesson 13 or later.
- No relevance threshold, hybrid search or reranker yet: nearest vectors can still be irrelevant.
- LLM-provided citations are **not** programmatically validated.
- Automated unit tests for `pdf_rag.py` have not been confirmed.
- `upsert()` doesn't automatically remove outdated records when re-chunking changes the number of chunks.
- Chunk ID scheme based on filename/page/chunk can collide for files with the same name; production ingestion needs document IDs/versioning.
- Some relative paths assume scripts are executed from repository root.

## Next lesson — Lesson 13: Advanced RAG

Suggested sequence:

1. Understand sparse keyword retrieval vs dense vector retrieval.
2. Implement a simple **BM25** search over the existing PDF chunks.
3. Compare BM25 and Chroma dense retrieval on realistic insurance questions.
4. Build **hybrid search**, merging lexical and semantic results (e.g., rank fusion).
5. Add metadata filters and relevance checks; understand limitations of raw distance thresholds.
6. Introduce a reranker, then compare top-k results.
7. Explore query rewriting and retrieval evaluation.
8. Integrate improved retrieval into `pdf_rag.py`, preserving real page citations and tests.

Follow current API documentation for any newly introduced library; avoid assuming exact package versions in this note.

## Common commands

```bash
cd ~/ai-engineering-journey
source .venv/bin/activate
python document_processing/pdf_chunker.py
python document_processing/ingest_pdf.py
python document_processing/pdf_rag.py
python -m unittest discover -s vector_databases -p 'test_chroma_rag.py' -v

git status
git add -A
git diff --cached --stat
git commit -m 'Complete Lesson 12 PDF ingestion and document RAG'
git push origin main
```

Before staging, verify `.gitignore` excludes `.venv/`, `.env`, cache folders and `vector_databases/chroma_data/`. Inspect `git diff --cached --name-only` to make sure no credentials, private documents or bulky generated data are included. Only use `git add -A` once the staged set is understood.

## Suggested message to start a new chat

> Continue my AI Engineering Journey at Lesson 13 — Advanced RAG. Please use the attached `LEARNING_PROGRESS.md` as the handover reference. I have completed Lessons 1–11 and implemented Lesson 12 using PyMuPDF, Ollama (`nomic-embed-text`, `qwen2:7b`), and ChromaDB (`insurance_policy_chunks`, 16 PDF chunks). I prefer step-by-step lessons with commented Python code, practical exercises, code review, minimal maths, and tests. Start with an explanation of BM25 versus semantic search, then a hands-on implementation using my existing insurance policy chunks. If you need exact source code, ask me to share the relevant files rather than guessing.
