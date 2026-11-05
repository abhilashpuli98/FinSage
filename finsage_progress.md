# finsage_progress.md — FinSage Extension Progress

## Current Status — 2026-11-05

- Phases 0–9 have implementation scaffolding in this checkout. Temporal paper extraction, evaluation models and runner, quality-cost benchmarking, figure generation, and end-to-end/ablation scripts are present; no external benchmark results or research conclusions are claimed.
- `ruff format --check .`, `ruff check .`, `basedpyright`, and `lint-imports` pass.
- The complete test suite passes: 704 tests and 142 subtests, 86.18% coverage, with three warnings.
- The router, conflict, and temporal-paper examples run in the project environment. No live model-provider evaluation was run for this status update.

## Completed Phases
- [x] Phase 6: TemporalPaperFlow implementation — flow, tests, and example are present
- [x] Phase 7: Evaluation scaffolding — benchmark models, constructor, evaluator, and comparison systems are present; benchmark results are not claimed
- [x] Phase 8: Quality-cost analysis scaffolding — model benchmark and figure-generation script are present; results are not claimed
- [x] Phase 9: End-to-end and ablation scaffolding — runner and tests are present; no live provider evaluation is claimed
<!-- Tick phases as implementation scaffolding is completed; research results require separate evidence. -->
- [x] Phase 0: Orientation — read AGENTS.md, contexts/, verify environment
- [x] Phase 1: Dependencies — pyproject.toml + import-linter update
- [x] Phase 2: TKU Models — knowledge/temporal.py + preprocess/time.py + tests
- [x] Phase 3: Conflict Models — knowledge/conflict.py + tests
- [x] Phase 4: LLM Router — configs/ollama.py + mind/router.py + tests
- [x] Phase 5: Conflict Detector — mind/conflict.py + tests

## Phase 2–5 Change Log
<!-- List every file touched, newest first -->
- `examples/mind/conflict.py` — runnable conflict detection + resolution example (created)
- `tests/mind/test_conflict.py` — 26 tests for detection, resolution, similarity, prefilter, and frozen-model immutability (created)
- `finsage/mind/conflict.py` — ConflictDetector, ConflictResolver, compute_cosine_similarity (created)
- `finsage/mind/__init__.py` — added conflict detection exports (modified)
- `examples/mind/router.py` — runnable default-routing example (created)
- `tests/mind/test_router.py` — 12 offline tests for routing, fallbacks, accounting, embeddings, and JSON parsing (created)
- `tests/configs/test_ollama.py` — 6 tests for Ollama and router configuration (created)
- `finsage/mind/router.py` — LiteLLM knowledge-pipeline router, cost tracker, and JSON parser (created)
- `finsage/mind/__init__.py` — added router exports (modified)
- `finsage/configs/ollama.py` — Ollama and routing configuration values (created)
- `finsage/configs/__init__.py` — added router configuration exports (modified)
- `finsage/knowledge/conflict.py` — ConflictRecord, ConflictType, ResolutionStatus (created)
- `tests/knowledge/test_conflict.py` — 29 tests for conflict models (created)
- `finsage/knowledge/temporal.py` — TemporalKnowledgeUnit, TKUCollection, FinanceDomain, KnowledgeType, TemporalDecay (created)
- `finsage/knowledge/__init__.py` — added temporal exports (modified)
- `finsage/preprocess/time.py` — added DateRange, extract_date_strings, normalize_date_range, infer_sample_period + helpers (modified)
- `finsage/preprocess/__init__.py` — added temporal exports (modified)
- `tests/knowledge/test_temporal.py` — 33 tests for TKU models (created)
- `tests/preprocess/test_time_temporal.py` — 38 tests for temporal time helpers (created)
- `finsage/evaluation/__init__.py` — placeholder for Phase 7 (created in Phase 1)
- `pyproject.toml` — FinSage deps + evaluation import-linter contract (modified in Phase 1)
- `finsage_progress.md` — this file (created)

## Historical Verification Checkpoints
- 2026-09-15 (Phase 5 complete):
  - `ruff format --check`: PASS
  - `ruff check`: PASS
  - `basedpyright`: PASS (0 errors, 0 warnings, 0 notes)
  - `lint-imports`: PASS (10/10 contracts KEPT, including mind → configs/knowledge/library/utils boundary)
  - Phase 0–5 regression suite: PASS (144 passed: knowledge/temporal, knowledge/conflict, preprocess/time_temporal, configs/ollama, mind/router, mind/conflict)
  - `python examples/mind/conflict.py`: PASS (detects 1 conflict, resolves via contextual synthesis)
  - `mind/conflict.py` coverage: 87% (branches + lines)
  - Full pytest remains blocked only by the previously recorded Windows-specific ETL surrogate-decoding and SQLite-handle cleanup failures outside FinSage.
- 2026-09-14 (Phase 4 audit):
  - `ruff format --check`: PASS (158 files)
  - `ruff check`: PASS
  - `basedpyright`: PASS (0 errors, 0 warnings, 0 notes)
  - `lint-imports`: PASS (10/10 contracts KEPT)
  - Phase 0–4 relevant regression suite: PASS (260 passed: knowledge, preprocess/time, configs, mind)
  - `python examples/mind/router.py`: PASS (`Summarization is routed to: local`)
  - Full pytest remains blocked only by the previously recorded Windows-specific ETL surrogate-decoding and SQLite-handle cleanup failures outside FinSage.
- 2026-09-14 (Phase 3 audit):
  - `ruff format --check`: PASS (153 files)
  - `ruff check`: PASS
  - `basedpyright`: PASS (0 errors, 0 warnings, 0 notes)
  - `lint-imports`: PASS (10/10 contracts KEPT, including evaluation apex)
  - Phase 2–3 functional tests: PASS (118 passed: temporal knowledge, preprocess time, conflict models)
  - Phase 1 dependency imports and Phase 2–3 package exports: PASS
  - Full `bash scripts/verify.sh`: unavailable because this Windows environment has no `/bin/bash`; its four static checks pass when run through `.venv\\Scripts`.
  - Full pytest: blocked by existing Windows-only test assumptions outside FinSage: `os.fsdecode(b"\\xff")` raises `UnicodeDecodeError` instead of producing a POSIX surrogate in ETL tests, and library tests retain SQLite handles while `TemporaryDirectory` attempts cleanup. No Phase 0–3 test failed.
- 2026-09-14 (Phase 2 complete):
  - `ruff format --check`: ✅ PASS (151 files)
  - `ruff check`: ✅ PASS (0 errors)
  - `basedpyright`: ✅ PASS (0 errors, 0 warnings, 0 notes)
  - `lint-imports`: ✅ PASS (10/10 contracts KEPT, 0 broken)
  - New tests: ✅ 71 passed (33 knowledge/temporal + 38 preprocess/time_temporal)
  - Regression: ✅ 133 passed (all existing knowledge + preprocess/time tests)

## Current Phase
The implementation scaffolding for Phases 0–9 is present. Further research work is to validate benchmark data and baselines, run controlled experiments, and report results with reproducible configurations.

## Notes / Blockers
- The historical Ollama startup issue below was recorded in September 2026 and should not be treated as a current environment status. Offline unit tests and mocked examples do not require a live Ollama server.

---

## Phase 0 — Orientation Answers

### What does `finsage/knowledge/` contain?
| File | Purpose |
|---|---|
| `__init__.py` | Public exports for all knowledge types |
| `_base.py` | `BaseKnowledge`, `Citation`, `SourceRef`, `ExtractionRef`, `ArtifactMeta` |
| `_flatten.py` | `FlattenKnowledge` shape |
| `_graph.py` | `GraphKnowledge` shape (placeholder) |
| `_tree.py` | `TreeNode`, `StructureTree`, `TreeKnowledge`, `StructureTreeValidationError` |
| `earnings.py` | `Earnings` (FlattenKnowledge subclass) |
| `factor.py` | `Factor` (FlattenKnowledge subclass) |
| `news.py` | `News` (FlattenKnowledge subclass) |
| `paper.py` | Full paper V1 model suite (~45 KB): `PaperSourceRevision`, `PaperChunkSet`, `PaperGlobalSummary`, `PaperStructureTree`, etc. |
| `thesis.py` | `Thesis` (FlattenKnowledge subclass) |

**Critical**: `knowledge/` is a **dependency leaf** — it CANNOT import from ANY other finsage module. New files here (e.g. `temporal.py`, `conflict.py`) must be pure Pydantic models with no LLM/external imports.

### What does `finsage/mind/` contain?
| File | Purpose |
|---|---|
| `__init__.py` | Exports `AgenticRetriever`, `Retrievable`, `RetrievalError`, `RetrievalEvidence` |
| `retrieval.py` | Pure-agentic reasoning retrieval over self-contained structure trees. Uses `agents` SDK directly. |

**Allowed imports**: `configs`, `knowledge`, `library`, `utils`. NO imports from `preprocess`, `rag`, `flows`, `etl`.

### What does `finsage/preprocess/` contain?
| File | Purpose |
|---|---|
| `__init__.py` | Public exports |
| `_news_types.py` | `NewsArtifact`, `NewsBatch`, `NewsDocument`, etc. (frozen dataclasses) |
| `clean.py` | `collapse_whitespace`, `dedupe_lines`, `normalize_unicode` |
| `fetch/` | HTTP fetchers, arXiv, RSS, DOI resolution |
| `format/` | PDF parsing, HTML→markdown |
| `news.py` | News preprocessing pipeline functions |
| `outline.py` | `HeadingCandidate`, `OutlineSignals`, `extract_outline_signals` |
| `time.py` | `to_utc`, `parse_filing_date`, `parse_news_datetime`, `business_days_between` |

**Allowed imports**: ONLY `finsage.utils`. NO configs, knowledge, mind, etc.
**Note**: `preprocess/time.py` already exists with time helpers. New temporal parsing for C1 (TKU) must be added here (extend, not replace).

### What does `finsage/flows/` contain?
| File | Purpose |
|---|---|
| `__init__.py` | Exports `PaperFlow`, `batch_run`, `BatchResult`, `collect_news`, errors |
| `_paper_summary.py` | Paper summarization internals |
| `_runner.py` | Flow runner utilities |
| `batch.py` | `batch_run` + `BatchResult` |
| `news.py` | `collect_news` |
| `paper/` | `PaperFlow` implementation + `_structure.py` |

**Apex layer**: Can import from everything (except transitional/deleted packages).

### What dependencies are already installed?
From `pyproject.toml [project] dependencies`:
- `arxiv`, `pyyaml`, `openai>=1.68.2`, `openai-agents>=0.14`
- `pillow>=10.1.0,<11.0.0`, `ipykernel>=6.29.5`
- `pydantic>=2.0.0`, `pymupdf>=1.23.0`, `python-dotenv>=1.0.0`
- `httpx[socks]>=0.28.1`, `litellm>=1.74.0.post1`
- `liteparse>=2.6,<3`, `llama-index-core>=0.14,<0.15`
- `llama-index-retrievers-bm25>=0.6,<0.7`
- `jinja2>=3.0.0`, `trafilatura>=1.10`

**Key**: `litellm` is already a core dependency — we use it for the LLM Router (C3).

### What optional extras exist?
- `[dev]`: ruff, basedpyright, import-linter, pytest, pytest-cov, pytest-asyncio, respx, pre-commit
- `[full]`: marker-pdf, beautifulsoup4, sentence-transformers

### Other modules
| Module | Files |
|---|---|
| `finsage/configs/` | `base.py`, `earnings.py`, `news.py`, `paper.py`, `retrieval.py`, `structure.py` |
| `finsage/library/` | `local.py`, `_types.py`, `_internal/` |
| `finsage/rag/` | `document.py` |
| `finsage/etl/` | `_pipeline.py`, `_batch.py`, `_record.py` (stdlib-only, independent leaf) |
| `finsage/utils/` | `logger.py`, `structured_output.py`, `usage.py` |
| `finsage/magic.py` | Natural language → `(input, cfg)` resolution |

### Import-linter contract summary
```
etl         → NO finsage imports at all (stdlib only)
utils       → NO finsage imports
knowledge   → NO finsage imports (CRITICAL: temporal.py, conflict.py must be pure Pydantic)
preprocess  → ONLY utils
configs     → ONLY knowledge
library     → ONLY knowledge
rag         → ONLY preprocess
mind        → ONLY configs, knowledge, library, utils
flows+magic → apex (no transitional packages: config, etl, flow, llm, models)
```

### Where new files go (FinSage contributions)
| Contribution | New File | Layer | Allowed Imports |
|---|---|---|---|
| C1: TKU | `knowledge/temporal.py` | leaf | none from finsage |
| C1: TKU | `preprocess/time.py` (extend) | layer 2 | only `utils` |
| C2: CSCG | `knowledge/conflict.py` | leaf | none from finsage |
| C2: CSCG | `mind/conflict.py` | layer 4 | configs, knowledge, library, utils |
| C3: Router | `configs/ollama.py` | layer 2 | only knowledge |
| C3: Router | `mind/router.py` | layer 4 | configs, knowledge, library, utils |
| C4: FinBench | `evaluation/` (NEW) | apex (5) | everything (needs import-linter contract) |
| Integration | `flows/temporal_paper.py` | apex (5) | everything |
