# Priqualis

**Pre-submission compliance validator for healthcare claim batches**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

Priqualis validates healthcare billing packages before submission to the payer, reducing rejections and accelerating reimbursement. It combines rule-based validation with hybrid similarity search to surface similar approved cases and generate safe auto-fix suggestions.

## Features

- **Rule Engine** — YAML-based DSL with three-state outcomes (SAT/VIOL/WARN) and impact scoring
- **Hybrid Similarity** — BM25 + vector ANN (HNSW) retrieval with optional cross-encoder re-rank
- **AutoFix** — Generates `patch.yaml` with auditable field-level corrections
- **Shadow Mode** — Import payer rejections to track First-Pass Acceptance (FPA) over time
- **PII Masking** — Regex/dictionary-based ETL ensures no sensitive data leaks

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  XML/CSV    │────▶│  ETL +      │────▶│  Rule Engine    │
│  Import     │     │  PII Mask   │     │  (YAML DSL)     │
└─────────────┘     └─────────────┘     └────────┬────────┘
                                                 │
                    ┌─────────────┐              ▼
                    │  AutoFix    │◀────┌─────────────────┐
                    │  Generator  │     │  Hybrid Search  │
                    └─────────────┘     │  BM25 + Vector  │
                                        └─────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Processing | Polars, DuckDB, Parquet |
| Search | Whoosh (BM25), Qdrant/pgvector (ANN) |
| Embeddings | e5-small / bge-small |
| API | FastAPI |
| UI | Streamlit |
| LLM | Explanation-only (no decisions) |

## Quick Start

```bash
# Clone and install
git clone https://github.com/SirSail/Priqualis.git
cd priqualis
pip install -e ".[dev]"

# Run API
uvicorn priqualis.api:app --reload

# Run UI
streamlit run Priqualis/ui/app.py
```

## Project Structure

```
soon
```

## Target KPIs

| Metric | Target |
|--------|--------|
| Formal error reduction | 20-30% |
| Violations with AutoFix | ≥40% |
| FPA improvement | +15-25 pp |
| 1k batch processing | ≤10-15 min |
| Similar query P95 | <300 ms |

## License

MIT

---

