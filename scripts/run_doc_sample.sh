#!/usr/bin/env bash
# Document-based extension on the synthetic sample document.
#   scripts/run_doc_sample.sh empty   # start from an empty ontology (M2-style)
#   scripts/run_doc_sample.sh base    # start from OntoDESIDE as the base ontology (M1-style)
# Requires a .env (see env_template) with OPENAI_API_KEY; set LLM_TEMPERATURE=0 for experiments.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-empty}"
DOC="${DOC:-$REPO/Dataset/sample_docs/textile_circularity_sample.md}"
RUN_DIR="${2:-$REPO/runs/doc_${MODE}_$(date +%Y%m%d_%H%M%S)}"
export LLM_COSTS_FILE="${LLM_COSTS_FILE:-$REPO/llm_costs.yaml}"
ONTO_ARGS=()
if [ "$MODE" = "base" ]; then
  ONTO_ARGS=(--onto-dir "$REPO/Dataset/OntoDESIDECoreOntology")
fi
cd "$REPO/OntoExtend Script"
python rag-ontoextend.py doc \
  --file "$DOC" \
  ${ONTO_ARGS[@]+"${ONTO_ARGS[@]}"} \
  --base-iri "http://example.org/doc-extension#" \
  --max-retries 2 \
  --top-k 20 \
  --run-dir "$RUN_DIR"
echo "Outputs: $RUN_DIR"
