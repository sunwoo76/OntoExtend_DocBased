"""Offline smoke test for rag-ontoextend.py (no API keys, no network calls).

Replaces the embedder with a deterministic hashing embedder and the LLM with a
stub that returns a small valid Turtle fragment per CQ, then runs the real
`OntologyRAG.run_csv()` pipeline end to end into a temporary --run-dir.

Usage (from repo root):
    python tests/smoke_offline.py [--limit 3] [--keep]

Checks:
  * reference ontologies are extracted and indexed
  * every CQ produces a fragment that passes validation
  * fragments from earlier CQs are retrievable by later CQs WITHOUT --include-cache
  * all outputs land in the run directory, not in ~/.core_ontology_rag_cache
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.util
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO / "OntoExtend Script"
DEFAULT_CQS = REPO / "Dataset" / "pre expriment data" / "RAG" / "testing-data" / "cqs.csv"
DEFAULT_ONTO = REPO / "Dataset" / "OntoDESIDECoreOntology"

# The OpenAI clients are constructed in __init__, so they need *some* key.
os.environ.setdefault("OPENAI_API_KEY", "offline-smoke-test")
os.environ["LLM_COSTS_FILE"] = str(REPO / "llm_costs.yaml")
sys.path.insert(0, str(SCRIPT_DIR))

# tiktoken downloads its BPE file on first use; offline we only need a stand-in
# (ENC is used solely as a token-count fallback for embedding cost logging).
import tiktoken  # noqa: E402

try:
    tiktoken.get_encoding("cl100k_base")
except Exception:
    class _ByteEncoding:
        def encode(self, text: str) -> List[int]:
            return list(text.encode("utf-8"))

    tiktoken.get_encoding = lambda name: _ByteEncoding()  # type: ignore[assignment]


def load_rag_module():
    spec = importlib.util.spec_from_file_location("rag_ontoextend", SCRIPT_DIR / "rag-ontoextend.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # needed by @dataclass during exec
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class HashEmbedder:
    """Bag-of-words feature hashing; deterministic and good enough for lexical overlap."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        tokens = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
        for t in tokens:
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v)[None, :]

    async def embed_one(self, text: str) -> np.ndarray:
        return self._vec(text)

    async def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        return [self._vec(t) for t in texts]


class StubLLM:
    """Returns one new class + one object property per CQ, always with domain/range."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: List[str] = []

    async def complete(self, prompt: str, system_prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        n = self.calls
        return (
            "```ttl\n"
            "@prefix : <http://example.org/smoke#> .\n"
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
            f":SmokeConcept{n} a owl:Class ; rdfs:label \"smoke concept {n} material product\" . # GENERATED\n"
            f":hasSmokePart{n} a owl:ObjectProperty ; rdfs:domain :SmokeConcept{n} ; rdfs:range :SmokeConcept{n} ;\n"
            f"    rdfs:label \"has smoke part {n}\" . # GENERATED\n"
            "```"
        )


async def run(limit: int, run_dir: Path) -> int:
    rag = load_rag_module()
    app = rag.OntologyRAG(rag.Paths(run_dir))
    app.embedder = HashEmbedder(int(rag.VECTOR_DIM))
    llm = StubLLM()
    app.llm = llm
    app.include_cache = False

    onto_files = rag.expand_onto_args(None, [str(DEFAULT_ONTO)])
    await app.run_csv(str(DEFAULT_CQS), onto_files, limit=limit, top_k=20)

    failures: List[str] = []
    frags = sorted((run_dir / "individual_ontologies").glob("*.ttl"))
    if len(frags) != limit:
        failures.append(f"expected {limit} fragments, found {len(frags)}")
    if not list(run_dir.glob("*_combined.ttl")):
        failures.append("combined ontology not written")
    with (run_dir / "cq_processing_log.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if [r["success"] for r in rows] != ["True"] * limit:
        failures.append(f"not all CQs succeeded: {[r['success'] for r in rows]}")
    # Session fragments are retrievable by later CQs even with include_cache=False.
    hits = await app._retrieve_relevant("smoke concept 1 material product", k=20)
    if not any(e.uri.endswith("SmokeConcept1") for e in hits):
        failures.append("fragment of CQ 1 is not retrievable after the run (include_cache=False)")
    if (Path.home() / ".core_ontology_rag_cache").exists() and not str(run_dir).startswith(str(Path.home() / ".core_ontology_rag_cache")):
        print("note: ~/.core_ontology_rag_cache exists (may predate this run)")

    print("\n=== smoke test ===")
    print(f"indexed elements: {len(app.store.elements)}; LLM calls: {llm.calls}; fragments: {len(frags)}")
    for msg in failures:
        print(f"FAIL: {msg}")
    print("PASS" if not failures else "FAILED")
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--keep", action="store_true", help="Keep the temporary run directory")
    args = ap.parse_args()
    run_dir = Path(tempfile.mkdtemp(prefix="ontoextend_smoke_"))
    try:
        code = asyncio.run(run(args.limit, run_dir))
    finally:
        if args.keep:
            print(f"outputs kept in {run_dir}")
        else:
            shutil.rmtree(run_dir, ignore_errors=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
