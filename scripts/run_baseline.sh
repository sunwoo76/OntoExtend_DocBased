#!/usr/bin/env bash
# Stage 0 baseline: original OntoExtend pipeline (M1) on the textile CQs with OntoDESIDE.
# Requires a .env (see env_template) with OPENAI_API_KEY; set LLM_TEMPERATURE=0 for experiments.
# Each run writes to its own directory so logs and cached fragments never mix between runs.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${1:-$REPO/runs/baseline_m1_textile_$(date +%Y%m%d_%H%M%S)}"
export LLM_COSTS_FILE="${LLM_COSTS_FILE:-$REPO/llm_costs.yaml}"  # default path is relative to cwd
cd "$REPO/OntoExtend Script"
python rag-ontoextend.py csv \
  --file "$REPO/Dataset/pre expriment data/RAG/testing-data/cqs.csv" \
  --onto-dir "$REPO/Dataset/OntoDESIDECoreOntology" \
  --top-k 20 \
  --run-dir "$RUN_DIR"
echo "Baseline outputs: $RUN_DIR"
