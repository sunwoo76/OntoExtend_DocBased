"""Offline tests for incremental ontology update and the document -> CQ pipeline.

No API keys or network: a hashing embedder and scripted LLM stubs stand in for OpenAI.
Run from the repo root:  python -m pytest tests -q
"""
from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import pytest
from rdflib import Graph, Namespace, OWL, RDF, RDFS, URIRef

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_offline as so  # noqa: E402  (sets env, stubs tiktoken, puts the script dir on sys.path)

rag = so.load_rag_module()
import doc_pipeline as dp  # noqa: E402

# Offline, tiktoken may be replaced by a byte counter (see smoke_offline); use word-based counts
# so chunk sizes in these tests match what the real tokenizer would roughly produce.
rag.ENC = type("ApproxEnc", (), {"encode": staticmethod(lambda t: [0] * dp.approx_tokens(t))})()

REPO = so.REPO
EX = "http://example.org/test#"
PREFIXES = (
    f"@prefix : <{EX}> .\n"
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
)


class ScriptedLLM:
    """Returns responses from a function of (prompt, system_prompt, call_number)."""

    def __init__(self, fn: Callable[[str, str, int], str]) -> None:
        self.fn = fn
        self.calls: List[tuple] = []

    async def complete(self, prompt: str, system_prompt: str) -> str:
        self.calls.append((prompt, system_prompt))
        return self.fn(prompt, system_prompt, len(self.calls))


def make_app(tmp_path: Path, llm, base_iri: str | None = EX, max_retries: int = 0):
    app = rag.OntologyRAG(rag.Paths(tmp_path))
    app.embedder = so.HashEmbedder(int(rag.VECTOR_DIM))
    app.llm = llm
    app.base_iri = base_iri
    app.max_retries = max_retries
    return app


def ttl(body: str, with_prefixes: bool = True) -> str:
    return "```ttl\n" + (PREFIXES if with_prefixes else "") + body + "\n```"


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- vector store

def test_vector_store_upsert_replaces_vector():
    store = rag.FaissVectorStore(4)
    e = rag.OntologyElement(uri="u:a", element_type="class", source_ontology="s")
    store.add(np.array([[1, 0, 0, 0]], "float32"), e)
    assert store.add(np.array([[0, 1, 0, 0]], "float32"), e) is False  # insert-once semantics kept
    assert store.upsert(np.array([[0, 1, 0, 0]], "float32"), e.model_copy(update={"label": "new"})) == "updated"
    assert len(store.elements) == 1 and store.idx.ntotal == 1
    (hit, score), = store.search(np.array([[0, 1, 0, 0]], "float32"), k=5)
    assert hit.label == "new" and score == pytest.approx(1.0)


# --------------------------------------------------------------------------- prefixes

def test_prefix_block_empty_start_and_base_iri(tmp_path):
    pm = rag.PrefixManager()
    assert f"@prefix : <{rag.DEFAULT_BASE_IRI}> ." in pm.build_prefix_block([])
    assert f"@prefix : <{EX}> ." in pm.build_prefix_block([], base_iri=EX)
    ref = tmp_path / "ref.ttl"
    ref.write_text("@prefix : <http://ref.org/o#> .\n@prefix owl: <http://www.w3.org/2002/07/owl#> .\n:A a owl:Class .\n")
    block = pm.build_prefix_block([str(ref)], base_iri=EX)
    pmap = pm.parse_prefix_block(block)
    assert pmap[""] == EX and pmap["core"] == "http://ref.org/o#"  # reference namespace stays addressable
    assert pm.parse_prefix_block(pm.build_prefix_block([str(ref)]))[""] == "http://ref.org/o#"  # M1 unchanged


# --------------------------------------------------------------------------- empty start + update

def test_empty_start_grows_and_updates_existing_elements(tmp_path):
    responses = {
        1: ttl(':Material a owl:Class ; rdfs:label "material" .\n'
               ':Fibre a owl:Class ; rdfs:label "fibre" .\n'
               ':hasFibre a owl:ObjectProperty ; rdfs:domain :Material ; rdfs:range :Fibre ; rdfs:label "has fibre" .'),
        # second CQ: reuses :Material, adds a restriction to it, and one new data property.
        # Written WITHOUT @prefix lines to exercise the prefix-block fallback.
        2: ttl(':Material a owl:Class ; rdfs:label "material" ;\n'
               '    rdfs:subClassOf [ a owl:Restriction ; owl:onProperty :hasFibre ; owl:someValuesFrom :Fibre ] .\n'
               ':fibreShare a owl:DatatypeProperty ; rdfs:domain :Material ; rdfs:range xsd:decimal ; rdfs:label "fibre share" .',
               with_prefixes=False),
        # third CQ repeats the same restriction: nothing new should be added for :Material
        3: ttl(':Material rdfs:subClassOf [ a owl:Restriction ; owl:onProperty :hasFibre ; owl:someValuesFrom :Fibre ] .'),
    }
    llm = ScriptedLLM(lambda p, s, n: responses[n])
    app = make_app(tmp_path, llm)

    async def go():
        await app.prepare([])
        assert len(app.store.elements) == 0
        o1 = await app.process_cq("What fibres does a material have?", "q1", cq_index=1)
        assert "(none yet" in llm.calls[0][0]  # bootstrap text when nothing can be retrieved
        o2 = await app.process_cq("What share of a material is a fibre?", "q2", cq_index=2)
        o3 = await app.process_cq("Which fibres must a material contain?", "q3", cq_index=3)
        return o1, o2, o3

    o1, o2, o3 = run(go())
    assert (o1.success, o1.new_elements, o1.changed_elements) == (True, 3, 0)
    assert (o2.success, o2.new_elements, o2.changed_elements) == (True, 1, 1)
    assert (o3.success, o3.new_elements, o3.changed_elements) == (True, 0, 0)

    # the store holds the UPDATED snippet of :Material (re-indexed, not skipped)
    mat = app.store.get(EX + "Material")
    assert mat is not None and "someValuesFrom" in (mat.snippet or "")
    assert len(app.store.elements) == 4

    # second prompt retrieved the elements created by the first CQ
    assert EX + "hasFibre" in llm.calls[1][0] or "hasFibre" in llm.calls[1][0]

    out = app.finalize("test")
    updated = Graph().parse(tmp_path / "test_updated.ttl")
    restrictions = list(updated.objects(URIRef(EX + "Material"), RDFS.subClassOf))
    assert len(restrictions) == 1, "repeated restriction must not be duplicated"
    assert (None, RDF.type, OWL.Ontology) in updated
    assert out is not None and Graph().parse(out)

    with (tmp_path / "element_provenance.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {(r["uri"].split("#")[-1], r["status"]) for r in rows} == {
        ("Material", "new"), ("Fibre", "new"), ("hasFibre", "new"),
        ("Material", "changed"), ("fibreShare", "new"),
    }
    with (tmp_path / "retrieved_elements" / "q2_retrieved.csv").open() as f:
        hdr = next(csv.reader(f))
    assert hdr[:2] == ["rank", "score"]


def test_with_base_ontology_extension_only_output(tmp_path):
    ref = tmp_path / "base.ttl"
    ref.write_text(PREFIXES + ':Product a owl:Class ; rdfs:label "product" .\n')
    llm = ScriptedLLM(lambda p, s, n: ttl(
        ':Product a owl:Class ; rdfs:label "product" .\n'
        ':Repairer a owl:Class ; rdfs:label "repairer" .\n'
        ':repairedBy a owl:ObjectProperty ; rdfs:domain :Product ; rdfs:range :Repairer .'))
    app = make_app(tmp_path / "run", llm, base_iri="http://example.org/ext#")

    async def go():
        await app.prepare([str(ref)])
        return await app.process_cq("Who repairs a product?", "q1", cq_index=1)

    o = run(go())
    assert o.success and o.new_elements == 2 and o.changed_elements == 0  # :Product only restated
    assert "put every NEW element" in llm.calls[0][0]
    app.finalize("t")
    ext = Graph().parse(tmp_path / "run" / "t_combined.ttl")
    assert (URIRef(EX + "Product"), RDFS.label, None) not in ext  # restated reference triples are not "extension"
    full = Graph().parse(tmp_path / "run" / "t_updated.ttl")
    assert (URIRef(EX + "Product"), RDF.type, OWL.Class) in full and (URIRef(EX + "repairedBy"), None, None) in full


def test_duplicate_label_candidates_logged(tmp_path):
    ref = tmp_path / "base.ttl"
    ref.write_text(PREFIXES + ':Material a owl:Class ; rdfs:label "material" .\n')
    llm = ScriptedLLM(lambda p, s, n: ttl(
        '@prefix ex2: <http://other.org/#> .\nex2:Material a owl:Class ; rdfs:label "Material" .'))
    app = make_app(tmp_path / "run", llm)
    run(app.prepare([str(ref)]))
    assert run(app.process_cq("What is a material?", "q1", cq_index=1)).success
    with (tmp_path / "run" / "duplicate_candidates.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["existing_uri"] == EX + "Material"


# --------------------------------------------------------------------------- self-correction

BAD = ttl(':hasPart a owl:ObjectProperty ; rdfs:label "has part" .')  # missing domain/range
GOOD = ttl(':Thing2 a owl:Class .\n:hasPart a owl:ObjectProperty ; rdfs:domain :Thing2 ; rdfs:range :Thing2 .')


@pytest.mark.parametrize("retries,success,calls", [(0, False, 1), (1, True, 2)])
def test_repair_loop(tmp_path, retries, success, calls):
    llm = ScriptedLLM(lambda p, s, n: BAD if n == 1 else GOOD)
    app = make_app(tmp_path, llm, max_retries=retries)
    run(app.prepare([]))
    o = run(app.process_cq("What are the parts of a thing?", "q1", cq_index=1))
    assert o.success is success and len(llm.calls) == calls
    if retries:
        assert "missing rdfs:domain and rdfs:range" in llm.calls[1][0]
        assert o.attempts == 2
    with (tmp_path / "cq_processing_log.csv").open() as f:
        row = list(csv.DictReader(f))[-1]
    assert row["success"] == str(success) and row["attempts"] == str(calls)


def test_invalid_turtle_is_a_failure_not_a_crash(tmp_path):
    app = make_app(tmp_path, ScriptedLLM(lambda p, s, n: "```ttl\n:A a owl:Class ; rdfs:label \"unterminated .\n```"))
    run(app.prepare([]))
    o = run(app.process_cq("What is A?", "q1", cq_index=1))
    assert not o.success and "Invalid Turtle" in (o.error or "")


# --------------------------------------------------------------------------- chunking

def test_chunk_markdown_headings_and_size():
    text = "# Doc\n\n## A\n\npara one.\n\npara two.\n\n## B\n\n" + ("word " * 300) + "\n\n```\n# not a heading\n```\n"
    chunks = dp.chunk_markdown(text, max_tokens=100)
    assert chunks[0].heading_path == ["Doc", "A"] and "para one." in chunks[0].text and "para two." in chunks[0].text
    b = [c for c in chunks if c.heading_path == ["Doc", "B"]]
    assert len(b) >= 3 and all(dp.approx_tokens(c.text) <= 160 for c in b)
    assert any("# not a heading" in c.text for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_overlap():
    paras = "\n\n".join(f"Paragraph {i} " + "x " * 30 for i in range(6))
    chunks = dp.chunk_markdown(paras, max_tokens=90, overlap_tokens=50)
    assert len(chunks) > 1
    last_para_first = chunks[0].text.split("\n\n")[-1]
    assert chunks[1].text.startswith(last_para_first)


def test_load_json_and_jsonl(tmp_path):
    j = tmp_path / "doc.json"
    j.write_text(json.dumps({"chunks": [{"id": "p1", "text": "Alpha.", "heading": "Intro", "page": 1},
                                        {"content": "Beta."}, "Gamma.", {"text": "  "}]}))
    cs = dp.load_chunks(str(j))
    assert [c.chunk_id for c in cs] == ["p1", "c0002", "c0003"]
    assert cs[0].heading_path == ["Intro"] and cs[0].page == "1"
    jl = tmp_path / "doc.jsonl"
    jl.write_text('{"text": "One."}\n{"markdown": "Two."}\n')
    assert [c.text for c in dp.load_chunks(str(jl))] == ["One.", "Two."]


# --------------------------------------------------------------------------- CQ generation

def test_grounding_and_json_parsing():
    chunk = "A blended material contains at least two different fibres. It is hard to recycle."
    assert dp.is_grounded("contains at least two different fibres", chunk)
    assert dp.is_grounded("A blended material contains at least two different fibres!", chunk)
    assert not dp.is_grounded("is made of recycled plastic bottles", chunk)
    assert dp.parse_cq_json('```json\n{"cqs": [{"cq": "What is X?", "evidence": "e"}]}\n```')[0]["cq"] == "What is X?"
    assert dp.parse_cq_json('Sure: {"cqs": ["Q1?"]} hope it helps') == [{"cq": "Q1?", "evidence": "", "type": ""}]
    assert dp.parse_cq_json("no json here") == []


def _doc_llm(prompt: str, system: str, n: int) -> str:
    """CQ-generation calls get JSON, extension calls get a fragment for the CQ."""
    if system == dp.CQ_SYS_PROMPT:
        if "Products and materials" in prompt:
            return json.dumps({"cqs": [
                {"cq": "What materials is a textile product made of?", "evidence": "A textile product is made of one or more materials."},
                {"cq": "What materials is a textile product made of", "evidence": "A textile product is made of one or more materials."},  # dup
                {"cq": "Which colour is a fibre?", "evidence": "Fibres are always blue."},  # ungrounded
            ]})
        if "Repair and maintenance" in prompt:
            return json.dumps({"cqs": [{"cq": "Who can repair a textile product?",
                                        "evidence": "A textile product can be repaired or maintained by a person."}]})
        return json.dumps({"cqs": []})
    cq = prompt.split('COMPETENCY QUESTION: "')[1].split('"')[0]
    if "materials" in cq:
        return ttl(':TextileProduct a owl:Class ; rdfs:label "textile product" .\n:Material a owl:Class .\n'
                   ':madeOf a owl:ObjectProperty ; rdfs:domain :TextileProduct ; rdfs:range :Material .')
    return ttl(':TextileProduct a owl:Class .\n:Person a owl:Class .\n'
               ':repairedBy a owl:ObjectProperty ; rdfs:domain :TextileProduct ; rdfs:range :Person .')


def test_document_pipeline_end_to_end(tmp_path):
    llm = ScriptedLLM(_doc_llm)
    app = make_app(tmp_path, llm)
    doc = REPO / "Dataset" / "sample_docs" / "textile_circularity_sample.md"
    run(app.run_document(str(doc), [], top_k=10, max_tokens=200))

    chunks = [json.loads(l) for l in (tmp_path / "chunks.jsonl").read_text().splitlines()]
    assert len(chunks) == 4 and chunks[0]["heading_path"][-1] == "1. Products and materials"
    with (tmp_path / "generated_cqs.csv").open() as f:
        rows = list(csv.DictReader(f))
    pairs = [(r["CQ"], r["status"]) for r in rows]
    assert pairs[:3] == [("What materials is a textile product made of?", "kept"),
                         ("What materials is a textile product made of?", "duplicate"),  # '?' normalised
                         ("Which colour is a fibre?", "ungrounded")]
    assert rows[1]["duplicate_of"] == rows[0]["CQID"]
    kept = [r for r in rows if r["status"] == "kept"]
    assert len(kept) == 2 and all(r["extension_success"] == "True" for r in kept)

    # the chunk text is given to the extension prompt as ONTOLOGY STORY
    ext_prompts = [p for p, s in llm.calls if s == rag.SYS_PROMPT]
    assert len(ext_prompts) == 2 and "ONTOLOGY STORY" in ext_prompts[0] and "Repair and maintenance" in ext_prompts[1]
    # the second CQ saw :TextileProduct created from the first chunk
    assert "TextileProduct" in ext_prompts[1].split("RELEVANT ONTOLOGY ELEMENTS (from")[1]

    summary = json.loads((tmp_path / "document_run_summary.json").read_text())
    assert summary["kept"] == 2 and summary["extended_ok"] == 2
    upd = Graph().parse(next(tmp_path.glob("*_updated.ttl")))
    assert (URIRef(EX + "repairedBy"), RDFS.range, URIRef(EX + "Person")) in upd
    with (tmp_path / "element_provenance.csv").open() as f:
        prov = list(csv.DictReader(f))
    assert {r["chunk_id"] for r in prov} == {"c0001", "c0003"}


def test_document_cq_only(tmp_path):
    llm = ScriptedLLM(_doc_llm)
    app = make_app(tmp_path, llm)
    doc = REPO / "Dataset" / "sample_docs" / "textile_circularity_sample.md"
    run(app.run_document(str(doc), [], cq_only=True))
    assert all(s == dp.CQ_SYS_PROMPT for _, s in llm.calls)
    assert not list(tmp_path.glob("*_updated.ttl"))
    with (tmp_path / "generated_cqs.csv").open() as f:
        assert sum(r["status"] == "kept" for r in csv.DictReader(f)) == 2


def test_cli_parses_doc_and_empty_onto(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "doc", "--file", "d.md", "--base-iri", EX, "--max-retries", "2"])
    a = rag.parse_args()
    assert a.cmd == "doc" and a.base_iri == EX and a.max_retries == 2 and a.chunk_tokens == 400
    assert rag.expand_onto_args(None, None) == []
