from __future__ import annotations
import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Literal

import faiss  # type: ignore
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from rdflib import BNode, Graph, Literal, Namespace, OWL, RDF, RDFS, URIRef
from rich import print
from tiktoken import get_encoding
from dotenv import load_dotenv
from llm_service import TokenCostTracker, get_llm_service
from embedder import get_embedder
import doc_pipeline

# ==========================
# Configuration & Constants
# ==========================

load_dotenv()  # take environment variables from .env

LLM_SERVICE = os.getenv("LLM_SERVICE", "openai")                    # openai or corporate
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))  # env values are strings

EMBEDDER = os.getenv("EMBEDDER", "openai")                          # openai or huggingface
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")    # embedding-3-small
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 1536))                              

ENC = get_encoding("cl100k_base")

# Namespace for new elements when neither --base-iri nor a reference default namespace is given.
DEFAULT_BASE_IRI = "http://example.org/ontoextend#"

STANDARD_PREFIXES: Dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

REQUIRED_PREFIXES: List[Tuple[str, str]] = [
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("bosch-meta-o", "http://semantic.bosch.com/bosch-meta-ontology/v1/"),
    ("bosch-metadata-o", "http://semantic.bosch.com/bosch-metadata-ontology-v1/"),
    ("brick", "https://brickschema.org/schema/Brick#"),
    ("cross-iso11179-registrationstatus-o", "http://semantic.bosch.com/cross-iso11179-registrationstatus-ontology-v1/"),
    ("csvw", "http://www.w3.org/ns/csvw#"),
    ("dc", "http://purl.org/dc/elements/1.1/"),
    ("dcam", "http://purl.org/dc/dcam/"),
    ("dcat", "http://www.w3.org/ns/dcat#"),
    ("dcmitype", "http://purl.org/dc/dcmitype/"),
    ("dct", "http://purl.org/dc/terms/"),
    ("doap", "http://usefulinc.com/ns/doap#"),
    ("edg", "http://edg.topbraid.solutions/model/"),
    ("foaf", "http://xmlns.com/foaf/0.1/"),
    ("geo", "http://www.opengis.net/ont/geosparql#"),
    ("metadata", "http://topbraid.org/metadata#"),
    ("odrl", "http://www.w3.org/ns/odrl/2/"),
    ("org", "http://www.w3.org/ns/org#"),
    ("prof", "http://www.w3.org/ns/dx/prof/"),
    ("prov", "http://www.w3.org/ns/prov#"),
    ("qb", "http://purl.org/linked-data/cube#"),
    ("schema", "https://schema.org/"),
    ("sh", "http://www.w3.org/ns/shacl#"),
    ("skos", "http://www.w3.org/2004/02/skos/core#"),
    ("sosa", "http://www.w3.org/ns/sosa/"),
    ("ssmo", "http://semantic.bosch.com/x-app-schema2semantic-mapping-ontology-v1/"),
    ("ssn", "http://www.w3.org/ns/ssn/"),
    ("teamwork", "http://topbraid.org/teamwork#"),
    ("time", "http://www.w3.org/2006/time#"),
    ("vann", "http://purl.org/vocab/vann/"),
    ("void", "http://rdfs.org/ns/void#"),
    ("wgs", "https://www.w3.org/2003/01/geo/wgs84_pos#"),
]

SH = Namespace("http://www.w3.org/ns/shacl#")

SYS_PROMPT = (
    "You are an ontology engineer. Reuse elements from the provided core ontology when possible. "
    "If you must create new elements, append '# GENERATED' as a comment. "
    "Return only Turtle/TTL syntax inside one markdown code block."
)

USER_TMPL = """
You are a helpful assistant designed to generate ontologies. You receive a COMPETENCY QUESTION (CQ) and optionally an ONTOLOGY STORY (OS). \n

{story_block}Based on CQ, which is a requirement for the ontology, and OS, which tells you what the context of the ontology is, your task is generating one ontology (O). The goal is to generate O that models the CQ properly. This means there is a way to write a SPARQL query to extract the answer to this CQ in O.  \n

Reuse the relevant ontology elements provided in the RELEVANT ONTOLOGY ELEMENTS section below whenever possible. These elements come from multiple reference ontologies. Only create new elements if absolutely necessary. In any case, add labels and comments. \n

Use the following prefixes:

{prefix_block}

 

Don't put any A-Box (instances) in the ontology and just generate the OWL file using Turtle syntax. Include the entities mentioned in the CQ. Remember to use restrictions when the CQ implies it. The output should be self-contained without any errors. Outside of the code box don't put any comment.\n

 

INSTRUCTIONS:

1. Analyze the CQ to understand what concepts and relationships are needed

2. Map the required concepts to classes/properties from the RELEVANT ONTOLOGY ELEMENTS above

3. Prefer elements with higher semantic similarity to the CQ concepts

4. Output only the TBox ontology in Turtle syntax (no instances/ABox)

5. Include restrictions (e.g., cardinality, value restrictions) when the CQ implies them. If the reference ontology uses OWL restrictions use only OWL restrictions; if the reference ontology only uses SHACL, then only use SHACL.

6. For every object or data property you introduce, include explicit rdfs:domain and rdfs:range statements (reuse the existing classes or datatypes when possible).

7. Ensure the output is syntactically correct and self-contained

 

COMPETENCY QUESTION: "{cq}"

 

RELEVANT ONTOLOGY ELEMENTS (from {num_ontologies} reference ontologies):

{relevant_elements}

 

NOTES:

- Do not use external ontologies. Only build on the ontologies provided in RELEVANT ONTOLOGY ELEMENTS .

- {namespace_note}

- Create the NodeShape with the same IRI as the Class it shapes, i.e. <ontology-prefix>:<ClassName> a sh:NodeShape .

- Create the PropertyShape names as follows: <ontology-prefix>:<ClassName>-<PropertyName> a sh:PropertyShape .

- For Property and Node Shapes: use the rdfs:label of the property or class as the sh:name .

- Every property definition must contain both rdfs:domain and rdfs:range statements.


- Do not create any new owl:Ontology statements .
"""

STORY_TMPL = """ONTOLOGY STORY (OS) -- the source text this CQ was derived from. Use it to understand the context and the intended meaning of the terms, but model only what the CQ needs:
\"\"\"
{story}
\"\"\"

"""

NO_ELEMENTS_TEXT = (
    "(none yet -- the ontology is empty or nothing relevant was found. Create the needed elements "
    "in the ':' namespace. Use UpperCamelCase for classes and lowerCamelCase for properties, and give "
    "each element an rdfs:label and rdfs:comment.)"
)

NAMESPACE_NOTE_REUSE = (
    'Reuse the namespaces (ontology prefix) used in the RELEVANT ONTOLOGY ELEMENTS whenever possible. '
    'Only if none are provided, use the ":" namespace .'
)
NAMESPACE_NOTE_BASE = (
    'Reuse existing elements with their own IRIs, but put every NEW element in the ":" namespace. '
    'Use UpperCamelCase for classes and lowerCamelCase for properties.'
)

REPAIR_TMPL = """Your previous answer could not be used because it failed validation.

ERROR:
{error}

PREVIOUS ANSWER:
```ttl
{fragment}
```

Fix the problem and return the complete corrected ontology fragment (not only the changed lines) in Turtle syntax inside one markdown code block. Keep the same modelling decisions unless the error requires changing them. Use the prefixes given in the original request.
"""

# ==========================
# Paths & IO
# ==========================

@dataclass
class Paths:
    cache_dir: Path = Path.home() / ".core_ontology_rag_cache"

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "individual_ontologies").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "retrieved_elements").mkdir(parents=True, exist_ok=True)

    @property
    def results_csv(self) -> Path:
        return self.cache_dir / "generation_runs.csv"

    @property
    def elements_csv(self) -> Path:
        return self.cache_dir / "ontology_elements.csv"

    @property
    def processing_log_csv(self) -> Path:
        return self.cache_dir / "cq_processing_log.csv"

    @property
    def token_cost_csv(self) -> Path:
        return self.cache_dir / "token_cost_log.csv"

    @property
    def combined_ttl(self) -> Path:
        return self.cache_dir / "combined_ontology.ttl"

    @property
    def provenance_csv(self) -> Path:
        return self.cache_dir / "element_provenance.csv"

    @property
    def duplicates_csv(self) -> Path:
        return self.cache_dir / "duplicate_candidates.csv"

    @property
    def working_ttl(self) -> Path:
        """Snapshot of the working ontology (reference ontologies + all merged fragments)."""
        return self.cache_dir / "working_ontology.ttl"

    @property
    def individual_dir(self) -> Path:
        return self.cache_dir / "individual_ontologies"

    @property
    def retrieved_dir(self) -> Path:
        return self.cache_dir / "retrieved_elements"


class RunLogger:
    def __init__(self, paths: Paths, tracker: TokenCostTracker) -> None:
        self.paths = paths
        self.tracker = tracker

    def log_elements(self, elements: List["OntologyElement"]) -> None:
        self.paths.elements_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.elements_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "uri", "element_type", "source_ontology", "label", "comment",
                "domain", "range", "super_classes", "sub_classes",
            ])
            ts = datetime.now().isoformat()
            for e in elements:
                w.writerow([
                    ts,
                    e.uri,
                    e.element_type,
                    Path(e.source_ontology).name,
                    e.label or "",
                    e.comment or "",
                    ";".join(e.domain or []),
                    ";".join(e.range or []),
                    ";".join(e.super_classes or []),
                    ";".join(e.sub_classes or []),
                ])

    def _append_rows(self, path: Path, header: List[str], rows: List[List[Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        newfile = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(header)
            w.writerows(rows)

    def log_provenance(self, cq_index: Any, basename: str, cq: str, chunk_id: Optional[str],
                       changes: List[Tuple[str, str, str, str]]) -> None:
        """changes: (uri, status new|changed, element_type, label)."""
        ts = datetime.now().isoformat()
        self._append_rows(
            self.paths.provenance_csv,
            ["timestamp", "cq_index", "fragment", "chunk_id", "cq", "uri", "status", "element_type", "label"],
            [[ts, cq_index, basename, chunk_id or "", cq, uri, status, etype, label]
             for uri, status, etype, label in changes],
        )

    def log_duplicates(self, cq_index: Any, basename: str, pairs: List[Tuple[str, str, str]]) -> None:
        """pairs: (new_uri, existing_uri, normalized_label)."""
        if not pairs:
            return
        ts = datetime.now().isoformat()
        self._append_rows(
            self.paths.duplicates_csv,
            ["timestamp", "cq_index", "fragment", "new_uri", "existing_uri", "normalized_label"],
            [[ts, cq_index, basename, a, b, lab] for a, b, lab in pairs],
        )

    def log_processing_result(
        self,
        cq_index: int,
        cq: str,
        success: bool,
        elements_used: int,
        source_ontologies: Dict[str, int],
        processing_time: float,
        error: Optional[str] = None,
        attempts: int = 1,
        new_elements: int = 0,
        changed_elements: int = 0,
        chunk_id: Optional[str] = None,
    ) -> None:
        self.paths.processing_log_csv.parent.mkdir(parents=True, exist_ok=True)
        newfile = not self.paths.processing_log_csv.exists()
        with self.paths.processing_log_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow([
                    "timestamp", "cq_index", "cq", "success", "elements_used",
                    "source_ontologies_used", "processing_time_seconds", "error",
                    "attempts", "new_elements", "changed_elements", "chunk_id",
                ])
            sources_str = ";".join([f"{Path(k).stem}:{v}" for k, v in source_ontologies.items()])
            w.writerow([
                datetime.now().isoformat(),
                cq_index,
                cq[:200] + "..." if len(cq) > 200 else cq,
                success,
                elements_used,
                sources_str,
                f"{processing_time:.2f}",
                error or "",
                attempts,
                new_elements,
                changed_elements,
                chunk_id or "",
            ])

    def log_final_results(
        self,
        total_cqs: int,
        successful: int,
        failed: int,
        total_time: float,
        output_file: str,
    ) -> None:
        self.paths.results_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.results_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "total_cqs", "successful", "failed", "success_rate_percent",
                "total_time_minutes", "avg_time_per_cq_seconds", "output_file",
            ])
            success_rate = (successful / total_cqs * 100) if total_cqs > 0 else 0
            avg_time = (total_time / total_cqs) if total_cqs > 0 else 0
            w.writerow([
                datetime.now().isoformat(),
                total_cqs,
                successful,
                failed,
                f"{success_rate:.1f}",
                f"{total_time / 60:.2f}",
                f"{avg_time:.2f}",
                output_file,
            ])

    def log_cost_row(self, cq_index: int, cq: str, deltas: Dict[str, Dict[str, int]]) -> None:
        self.paths.token_cost_csv.parent.mkdir(parents=True, exist_ok=True)
        newfile = not self.paths.token_cost_csv.exists()
        with self.paths.token_cost_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow([
                    "timestamp", "cq_index", "cq", "model",
                    "prompt_tokens", "completion_tokens", "embedding_tokens", "cost_usd",
                ])
            for model, counts in deltas.items():
                cost = self.tracker.model_cost_usd(model, counts)
                w.writerow([
                    datetime.now().isoformat(),
                    cq_index,
                    cq[:200] + ("..." if len(cq) > 200 else ""),
                    model,
                    counts.get("prompt", 0),
                    counts.get("completion", 0),
                    counts.get("embedding", 0),
                    f"{cost:.6f}",
                ])


# ==========================
# Ontology Elements
# ==========================

class OntologyElement(BaseModel):
    uri: str
    element_type: str  # 'class', 'object_property', 'data_property', 'annotation_property'
    source_ontology: str
    label: Optional[str] = None
    comment: Optional[str] = None
    domain: Optional[List[str]] = Field(default_factory=list)
    range: Optional[List[str]] = Field(default_factory=list)
    super_classes: Optional[List[str]] = Field(default_factory=list)
    sub_classes: Optional[List[str]] = Field(default_factory=list)
    additional_info: Optional[List[str]] = Field(default_factory=list)
    snippet: Optional[str] = None

    def get_searchable_text(self) -> str:
        parts: List[str] = []
        local_name = self.uri.split('#')[-1].split('/')[-1]
        parts.append(local_name)
        if self.label:
            parts.append(self.label)
        if self.comment:
            parts.append(self.comment)
        parts.append(f"Type: {self.element_type.replace('_', ' ')}")
        if self.domain:
            domain_names = [d.split('#')[-1].split('/')[-1] for d in self.domain]
            parts.append(f"Domain: {', '.join(domain_names)}")
        if self.range:
            range_names = [r.split('#')[-1].split('/')[-1] for r in self.range]
            parts.append(f"Range: {', '.join(range_names)}")
        if self.additional_info:
            parts.extend(self.additional_info)
        return " | ".join(parts)


# ==========================
# Prefix Management
# ==========================

class PrefixManager:
    def __init__(self) -> None:
        pass

    @staticmethod
    def sanitize_prefix_name(raw: Optional[str]) -> str:
        s = (raw or "").strip()
        s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
        if not s or not re.match(r"^[A-Za-z_]", s):
            s = f"ns_{s}" if s else "ns"
        s = re.sub(r"_+", "_", s)
        return s

    def collect_prefixes_from_files(self, ontology_files: List[str]) -> Tuple[Optional[str], Dict[str, str]]:
        extra_prefixes: Dict[str, str] = {}
        used: set[str] = set(STANDARD_PREFIXES.keys())
        default_ns: Optional[str] = None
        for f in ontology_files:
            g = Graph()
            try:
                g.parse(f)
            except Exception:
                continue
            for pref, ns in g.namespaces():
                ns = str(ns)
                if not ns:
                    continue
                pref_str = "" if pref is None else str(pref)
                if pref_str.lower() == "xml":
                    continue
                if pref_str == "":
                    if not default_ns:
                        default_ns = ns
                    continue
                if pref_str in STANDARD_PREFIXES:
                    continue
                p = self.sanitize_prefix_name(pref_str)
                while p in used:
                    m = re.search(r"(\d+)$", p)
                    p = re.sub(r"\d+$", str(int(m.group(1)) + 1), p) if m else f"{p}2"
                extra_prefixes[p] = ns
                used.add(p)
        return default_ns, extra_prefixes

    def build_prefix_block(self, ontology_files: List[str], base_iri: Optional[str] = None) -> str:
        """Build the @prefix block given to the LLM.

        ':' is bound to `base_iri` when given, otherwise to the reference ontologies' default
        namespace, otherwise to DEFAULT_BASE_IRI (so an empty start always has a ':' prefix).
        When `base_iri` displaces a reference default namespace, that namespace keeps a named
        prefix ('core', 'core2', ...) so its elements stay addressable.
        """
        default_ns, extra_prefixes = self.collect_prefixes_from_files(ontology_files)
        lines: List[str] = []
        used: Set[str] = set()
        colon_ns = base_iri or default_ns or DEFAULT_BASE_IRI
        lines.append(f"@prefix : <{colon_ns}> .")
        used.add("")
        if default_ns and default_ns != colon_ns and default_ns not in extra_prefixes.values():
            name = "core"
            while name in extra_prefixes or name in STANDARD_PREFIXES:
                m = re.search(r"(\d+)$", name)
                name = re.sub(r"\d+$", str(int(m.group(1)) + 1), name) if m else f"{name}2"
            extra_prefixes[name] = default_ns
        for pref, uri in REQUIRED_PREFIXES:
            if pref in used:
                continue
            lines.append(f"@prefix {pref}: <{uri}> .")
            used.add(pref)
        for pref in sorted(extra_prefixes.keys(), key=str.lower):
            if pref in used:
                continue
            lines.append(f"@prefix {pref}: <{extra_prefixes[pref]}> .")
            used.add(pref)
        return "\n".join(lines) + "\n"

    @staticmethod
    def parse_prefix_block(prefix_block: str) -> Dict[str, str]:
        return {m.group(1): m.group(2) for m in re.finditer(r"@prefix\s+([A-Za-z0-9_\-]*):\s*<([^>]*)>", prefix_block)}


# ==========================
# Extraction
# ==========================

class OntologyExtractor:
    def __init__(self, logger: RunLogger) -> None:
        self.logger = logger

    @staticmethod
    def _collect_snippet(g: Graph, subject: URIRef, depth: int = 1) -> str:
        snippet = Graph()
        for pref, ns in g.namespaces():
            if pref is None:
                snippet.bind('', ns)
            else:
                snippet.bind(pref, ns)

        visited_bnodes: Set[BNode] = set()

        def add_triples(node: Any, remaining: int) -> None:
            for pred, obj in g.predicate_objects(node):
                snippet.add((node, pred, obj))
                if remaining > 0 and isinstance(obj, BNode) and obj not in visited_bnodes:
                    visited_bnodes.add(obj)
                    add_triples(obj, remaining - 1)

        add_triples(subject, depth)
        try:
            return snippet.serialize(format="turtle").strip()
        except Exception:
            return ""

    def extract_from_files(self, ontology_files: List[str], *, log: bool = True) -> List[OntologyElement]:
        print(f"[blue]Loading {len(ontology_files)} reference ontologies...[/blue]")
        all_elements: List[OntologyElement] = []
        for ontology_file in ontology_files:
            p = Path(ontology_file)
            if not p.exists():
                print(f"[red]Ontology file not found: {ontology_file}[/red]")
                continue
            print(f"[cyan]  Processing: {p.name}[/cyan]")
            try:
                elements = self._extract_single(str(p))
                all_elements.extend(elements)
                print(f"[green] Extracted {len(elements)} elements[/green]")
            except Exception as e:
                print(f"[red]Error processing {ontology_file}: {e}[/red]")
                continue
        print(f"[green]✅ Total extracted {len(all_elements)} elements from {len(ontology_files)} ontologies[/green]")
        if log:
            self.logger.log_elements(all_elements)
        return all_elements

    def _extract_single(self, ontology_file: str) -> List[OntologyElement]:
        g = Graph()
        g.parse(ontology_file)
        return self.extract_from_graph(g, ontology_file)

    def extract_from_graph(self, g: Graph, source: str, subjects: Optional[Set[URIRef]] = None) -> List[OntologyElement]:
        """Extract elements from an in-memory graph; if `subjects` is given, only those IRIs."""
        def keep(node: Any) -> bool:
            return subjects is None or node in subjects

        elements: List[OntologyElement] = []
        ontology_file = source
        # classes
        for cls in list(g.subjects(RDF.type, OWL.Class)):
            if isinstance(cls, URIRef) and keep(cls):
                el = self._extract_class_info(g, cls, ontology_file)
                if el:
                    elements.append(el)
        # object properties
        for prop in list(g.subjects(RDF.type, OWL.ObjectProperty)):
            if isinstance(prop, URIRef) and keep(prop):
                el = self._extract_property_info(g, prop, 'object_property', ontology_file)
                if el:
                    elements.append(el)
        # data properties
        for prop in list(g.subjects(RDF.type, OWL.DatatypeProperty)):
            if isinstance(prop, URIRef) and keep(prop):
                el = self._extract_property_info(g, prop, 'data_property', ontology_file)
                if el:
                    elements.append(el)
        # annotation properties
        for prop in list(g.subjects(RDF.type, OWL.AnnotationProperty)):
            if isinstance(prop, URIRef) and keep(prop):
                el = self._extract_property_info(g, prop, 'annotation_property', ontology_file)
                if el:
                    elements.append(el)

        # SHACL NodeShapes
        for shape in list(g.subjects(RDF.type, SH.NodeShape)):
            if not keep(shape):
                continue
            el = self._extract_shacl_shape(g, shape, ontology_file, 'node_shape')
            if el:
                elements.append(el)

        # SHACL PropertyShapes
        for shape in list(g.subjects(RDF.type, SH.PropertyShape)):
            if not keep(shape):
                continue
            el = self._extract_shacl_shape(g, shape, ontology_file, 'property_shape')
            if el:
                elements.append(el)

        # Named OWL restrictions (skip blank nodes)
        for restriction in list(g.subjects(RDF.type, OWL.Restriction)):
            if not keep(restriction):
                continue
            el = self._extract_named_restriction(g, restriction, ontology_file)
            if el:
                elements.append(el)
        return elements

    @staticmethod
    def _extract_class_info(g: Graph, cls: URIRef, source_ontology: str) -> Optional[OntologyElement]:
        try:
            label = OntologyExtractor._get_literal_value(g, cls, RDFS.label)
            comment = OntologyExtractor._get_literal_value(g, cls, RDFS.comment)
            super_classes = [str(sc) for sc in g.objects(cls, RDFS.subClassOf) if isinstance(sc, URIRef)]
            sub_classes = [str(sc) for sc in g.subjects(RDFS.subClassOf, cls) if isinstance(sc, URIRef)]
            return OntologyElement(
                uri=str(cls),
                element_type='class',
                source_ontology=source_ontology,
                label=label,
                comment=comment,
                super_classes=super_classes,
                sub_classes=sub_classes,
                snippet=OntologyExtractor._collect_snippet(g, cls),
            )
        except Exception as e:
            print(f"[yellow] Error extracting class {cls}: {e}[/yellow]")
            return None

    @staticmethod
    def _extract_property_info(g: Graph, prop: URIRef, prop_type: str, source_ontology: str) -> Optional[OntologyElement]:
        try:
            label = OntologyExtractor._get_literal_value(g, prop, RDFS.label)
            comment = OntologyExtractor._get_literal_value(g, prop, RDFS.comment)
            domain = [str(d) for d in g.objects(prop, RDFS.domain) if isinstance(d, URIRef)]
            range_vals = [str(r) for r in g.objects(prop, RDFS.range) if isinstance(r, URIRef)]
            return OntologyElement(
                uri=str(prop),
                element_type=prop_type,
                source_ontology=source_ontology,
                label=label,
                comment=comment,
                domain=domain,
                range=range_vals,
                snippet=OntologyExtractor._collect_snippet(g, prop),
            )
        except Exception as e:
            print(f"[yellow]Error extracting property {prop}: {e}[/yellow]")
            return None

    @staticmethod
    def _get_literal_value(g: Graph, subject: URIRef, predicate: URIRef) -> Optional[str]:
        for obj in g.objects(subject, predicate):
            if isinstance(obj, Literal):
                return str(obj)
        return None

    @staticmethod
    def _local_name(uri: str) -> str:
        return uri.split('#')[-1].split('/')[-1]

    @staticmethod
    def _format_uri_list(values: List[str]) -> str:
        return ", ".join(OntologyExtractor._local_name(v) for v in values)

    @staticmethod
    def _extract_shacl_shape(g: Graph, shape: Any, source_ontology: str, shape_type: str) -> Optional[OntologyElement]:
        if not isinstance(shape, URIRef):
            return None
        label = OntologyExtractor._get_literal_value(g, shape, RDFS.label)
        comment = OntologyExtractor._get_literal_value(g, shape, RDFS.comment)
        info: List[str] = []
        target_classes = [str(tc) for tc in g.objects(shape, SH.targetClass) if isinstance(tc, URIRef)]
        if target_classes:
            info.append(f"Target Classes: {OntologyExtractor._format_uri_list(target_classes)}")
        target_props = [str(tp) for tp in g.objects(shape, SH.targetObjectsOf) if isinstance(tp, URIRef)]
        if target_props:
            info.append(f"Target Objects Of: {OntologyExtractor._format_uri_list(target_props)}")
        for prop in g.objects(shape, SH.property):
            if isinstance(prop, URIRef):
                info.append(f"Uses Property Shape: {OntologyExtractor._local_name(str(prop))}")
        path = next((str(p) for p in g.objects(shape, SH.path) if isinstance(p, URIRef)), None)
        if path:
            info.append(f"Path: {OntologyExtractor._local_name(path)}")
        datatype = next((str(dt) for dt in g.objects(shape, SH.datatype) if isinstance(dt, URIRef)), None)
        if datatype:
            info.append(f"Datatype: {OntologyExtractor._local_name(datatype)}")
        klass = next((str(cl) for cl in g.objects(shape, SH.class_) if isinstance(cl, URIRef)), None)
        if klass:
            info.append(f"Value Class: {OntologyExtractor._local_name(klass)}")
        min_count = OntologyExtractor._get_literal_value(g, shape, SH.minCount)
        if min_count:
            info.append(f"Min Count: {min_count}")
        max_count = OntologyExtractor._get_literal_value(g, shape, SH.maxCount)
        if max_count:
            info.append(f"Max Count: {max_count}")
        return OntologyElement(
            uri=str(shape),
            element_type=shape_type,
            source_ontology=source_ontology,
            label=label,
            comment=comment,
            additional_info=info,
            snippet=OntologyExtractor._collect_snippet(g, shape),
        )

    @staticmethod
    def _extract_named_restriction(g: Graph, restriction: Any, source_ontology: str) -> Optional[OntologyElement]:
        if not isinstance(restriction, URIRef) or isinstance(restriction, BNode):
            return None
        on_property = next((str(op) for op in g.objects(restriction, OWL.onProperty) if isinstance(op, URIRef)), None)
        some_values = next((str(val) for val in g.objects(restriction, OWL.someValuesFrom) if isinstance(val, URIRef)), None)
        all_values = next((str(val) for val in g.objects(restriction, OWL.allValuesFrom) if isinstance(val, URIRef)), None)
        has_value = OntologyExtractor._get_literal_value(g, restriction, OWL.hasValue)
        info: List[str] = []
        if on_property:
            info.append(f"On Property: {OntologyExtractor._local_name(on_property)}")
        if some_values:
            info.append(f"Some Values From: {OntologyExtractor._local_name(some_values)}")
        if all_values:
            info.append(f"All Values From: {OntologyExtractor._local_name(all_values)}")
        if has_value:
            info.append(f"Has Value: {has_value}")
        for pred, label in [
            (OWL.cardinality, "Cardinality"),
            (OWL.minCardinality, "Min Cardinality"),
            (OWL.maxCardinality, "Max Cardinality"),
            (OWL.qualifiedCardinality, "Qualified Cardinality"),
            (OWL.minQualifiedCardinality, "Min Qualified Cardinality"),
            (OWL.maxQualifiedCardinality, "Max Qualified Cardinality"),
        ]:
            val = OntologyExtractor._get_literal_value(g, restriction, pred)
            if val:
                info.append(f"{label}: {val}")
        label_literal = OntologyExtractor._get_literal_value(g, restriction, RDFS.label)
        comment = OntologyExtractor._get_literal_value(g, restriction, RDFS.comment)
        return OntologyElement(
            uri=str(restriction),
            element_type='restriction',
            source_ontology=source_ontology,
            label=label_literal,
            comment=comment,
            additional_info=info,
            snippet=OntologyExtractor._collect_snippet(g, restriction),
        )


class FaissVectorStore:
    """FAISS inner-product index keyed by element URI.

    Uses IndexIDMap2 so that a URI's vector can be replaced when the element changes
    (e.g. a later fragment adds axioms to an existing class): `upsert()` swaps the vector
    and the element in place, `add()` keeps the original insert-once behaviour.
    """

    def __init__(self, dim: int, inner_product: bool = True) -> None:
        base = faiss.IndexFlatIP(dim) if inner_product else faiss.IndexFlatL2(dim)
        self.idx = faiss.IndexIDMap2(base)
        self._by_id: Dict[int, OntologyElement] = {}
        self._id_by_uri: Dict[str, int] = {}
        self._next_id = 0

    @property
    def elements(self) -> List[OntologyElement]:
        return list(self._by_id.values())

    @property
    def seen_uris(self) -> Set[str]:
        return set(self._id_by_uri.keys())

    def is_new(self, uri: str) -> bool:
        return uri not in self._id_by_uri

    def get(self, uri: str) -> Optional[OntologyElement]:
        i = self._id_by_uri.get(uri)
        return self._by_id.get(i) if i is not None else None

    def _insert(self, vec: np.ndarray, element: OntologyElement) -> None:
        i = self._next_id
        self._next_id += 1
        self.idx.add_with_ids(np.asarray(vec, dtype="float32").reshape(1, -1), np.array([i], dtype="int64"))
        self._by_id[i] = element
        self._id_by_uri[element.uri] = i

    def add(self, vec: np.ndarray, element: OntologyElement) -> bool:
        if not element.uri or element.uri in self._id_by_uri:
            return False
        self._insert(vec, element)
        return True

    def upsert(self, vec: np.ndarray, element: OntologyElement) -> str:
        """Insert or replace; returns 'added' or 'updated'."""
        if not element.uri:
            return "skipped"
        old = self._id_by_uri.pop(element.uri, None)
        if old is not None:
            self.idx.remove_ids(np.array([old], dtype="int64"))
            self._by_id.pop(old, None)
        self._insert(vec, element)
        return "updated" if old is not None else "added"

    def search(self, vec: np.ndarray, k: int = 10) -> List[Tuple[OntologyElement, float]]:
        if self.idx.ntotal == 0:
            return []
        D, I = self.idx.search(np.asarray(vec, dtype="float32").reshape(1, -1), min(k, self.idx.ntotal))
        return [(self._by_id[int(i)], float(D[0][j])) for j, i in enumerate(I[0]) if i != -1 and int(i) in self._by_id]


# ==========================
# Working ontology (incremental update)
# ==========================

@dataclass
class MergeResult:
    new_subjects: Set[URIRef]
    changed_subjects: Set[URIRef]
    added_triples: int
    duplicate_candidates: List[Tuple[str, str, str]]


def _normalize_label(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class WorkingOntology:
    """One rdflib graph that grows with every accepted fragment.

    `base` holds the reference (starting) ontology, `extension` holds only what the
    fragments added. Merging is done on triples, so a fragment that re-declares an
    existing class only contributes the axioms that are actually new.
    """

    def __init__(self, prefix_map: Dict[str, str], ontology_iri: str) -> None:
        self.prefix_map = prefix_map
        self.ontology_iri = ontology_iri
        self.base = Graph()
        self.extension = Graph()
        for g in (self.base, self.extension):
            self._bind(g)

    def _bind(self, g: Graph) -> None:
        for pref, ns in self.prefix_map.items():
            g.bind(pref, Namespace(ns), override=True, replace=True)

    def load_base(self, files: List[str]) -> None:
        for f in files:
            try:
                self.base.parse(f)
            except Exception as e:
                print(f"[yellow]Could not load {f} into the working ontology: {e}[/yellow]")
        # remove owl:Ontology headers of the inputs; the outputs get a single header
        for s in list(self.base.subjects(RDF.type, OWL.Ontology)):
            self.base.remove((s, None, None))
        self._bind(self.base)

    def full(self) -> Graph:
        g = Graph()
        self._bind(g)
        for t in self.base:
            g.add(t)
        for t in self.extension:
            g.add(t)
        return g

    def _labels(self) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for g in (self.base, self.extension):
            for s, o in g.subject_objects(RDFS.label):
                if isinstance(s, URIRef):
                    out.setdefault(_normalize_label(str(o)), set()).add(str(s))
            for s in g.subjects(RDF.type, None):
                if isinstance(s, URIRef):
                    out.setdefault(_normalize_label(OntologyExtractor._local_name(str(s))), set()).add(str(s))
        return out

    @staticmethod
    def _bnode_signature(g: Graph, node: Any, depth: int = 12) -> Any:
        """Structural signature of a blank-node subtree (restrictions, lists, unions...)."""
        if not isinstance(node, BNode):
            return node
        if depth <= 0:
            return "..."
        return tuple(sorted(
            (str(p), repr(WorkingOntology._bnode_signature(g, o, depth - 1)))
            for p, o in g.predicate_objects(node)
        ))

    @staticmethod
    def _bnode_closure(g: Graph, node: Any) -> Set[BNode]:
        seen: Set[BNode] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if not isinstance(n, BNode) or n in seen:
                continue
            seen.add(n)
            stack.extend(o for o in g.objects(n, None))
        return seen

    def merge(self, fragment: Graph) -> MergeResult:
        for s in list(fragment.subjects(RDF.type, OWL.Ontology)):
            fragment.remove((s, None, None))
        before = self.full()
        known = set(before.subjects())
        labels_before = self._labels()

        # Blank-node structures (restrictions etc.) get fresh ids on every parse, so compare
        # them structurally: a fragment that repeats an existing restriction adds nothing.
        skip_triples: Set[Tuple[Any, Any, Any]] = set()
        skip_bnodes: Set[BNode] = set()
        for s, p, o in fragment:
            if isinstance(s, URIRef) and isinstance(o, BNode):
                sig = self._bnode_signature(fragment, o)
                existing = {self._bnode_signature(before, o2) for o2 in before.objects(s, p) if isinstance(o2, BNode)}
                if sig in existing:
                    skip_triples.add((s, p, o))
                    skip_bnodes |= self._bnode_closure(fragment, o)

        new_subjects: Set[URIRef] = set()
        changed: Set[URIRef] = set()
        added = 0
        added_bnodes: Set[BNode] = set()
        for t in fragment:
            if t in skip_triples or t[0] in skip_bnodes or t in before:
                continue
            self.extension.add(t)
            added += 1
            s = t[0]
            if isinstance(s, URIRef):
                (changed if s in known else new_subjects).add(s)
            elif isinstance(s, BNode):
                added_bnodes.add(s)
        # A new blank-node axiom belongs to the named node(s) that point to it.
        if added_bnodes:
            full = self.full()
            for b in added_bnodes:
                for owner in self._owners(full, b):
                    if owner not in new_subjects:
                        (changed if owner in known else new_subjects).add(owner)

        dups: List[Tuple[str, str, str]] = []
        for s in new_subjects:
            names = {_normalize_label(OntologyExtractor._local_name(str(s)))}
            names |= {_normalize_label(str(o)) for o in fragment.objects(s, RDFS.label)}
            for n in names:
                for other in labels_before.get(n, set()):
                    if n and other != str(s):
                        dups.append((str(s), other, n))
        return MergeResult(new_subjects, changed, added, sorted(set(dups)))

    @staticmethod
    def _owners(g: Graph, node: Any, depth: int = 4) -> Set[URIRef]:
        owners: Set[URIRef] = set()
        frontier = {node}
        for _ in range(depth):
            nxt = set()
            for n in frontier:
                for s in g.subjects(None, n):
                    if isinstance(s, URIRef):
                        owners.add(s)
                    elif isinstance(s, BNode):
                        nxt.add(s)
            if not nxt:
                break
            frontier = nxt
        return owners

    def _with_header(self, g: Graph, comment: str) -> Graph:
        onto = URIRef(self.ontology_iri)
        g.add((onto, RDF.type, OWL.Ontology))
        g.add((onto, RDFS.comment, Literal(comment)))
        return g

    def serialize_updated(self, path: Path) -> None:
        g = self._with_header(self.full(), "Reference ontology extended with fragments generated from competency questions (OntoExtend).")
        path.write_text(g.serialize(format="turtle"), encoding="utf-8")

    def serialize_extension(self, path: Path, reference_files: List[str], n_fragments: int) -> None:
        g = Graph()
        self._bind(g)
        for t in self.extension:
            g.add(t)
        refs = ", ".join(Path(f).name for f in reference_files) or "none (started from an empty ontology)"
        self._with_header(g, f"Extension only: {n_fragments} merged fragments. Reference ontologies: {refs}.")
        path.write_text(g.serialize(format="turtle"), encoding="utf-8")


# ==========================
# Prompting & LLM
# ==========================

class PromptBuilder:
    def __init__(self, prefix_block: str, new_elements_in_base: bool = False) -> None:
        self.prefix_block = prefix_block
        self.namespace_note = NAMESPACE_NOTE_BASE if new_elements_in_base else NAMESPACE_NOTE_REUSE

    @staticmethod
    def _format_relevant_elements(elements: List[OntologyElement]) -> Tuple[str, int]:
        elements_by_source: Dict[str, List[OntologyElement]] = {}
        for e in elements:
            src = Path(e.source_ontology).stem
            elements_by_source.setdefault(src, []).append(e)
        parts: List[str] = []
        for source, els in elements_by_source.items():
            parts.append(f"\n## From {source} ontology:")
            for e in els:
                parts.append(f"- {e.get_searchable_text()}")
                if e.snippet:
                    snippet = e.snippet.strip()
                    if snippet:
                        parts.append("```ttl")
                        parts.append(snippet)
                        parts.append("```")
        return "\n".join(parts), len(elements_by_source)

    def build(self, cq: str, elements: List[OntologyElement], story: Optional[str] = None) -> str:
        relevant_text, num_ontologies = self._format_relevant_elements(elements)
        if not elements:
            relevant_text = NO_ELEMENTS_TEXT
        story_block = STORY_TMPL.format(story=story.strip()) if story and story.strip() else ""
        return USER_TMPL.format(
            num_ontologies=num_ontologies,
            relevant_elements=relevant_text,
            cq=cq,
            prefix_block=self.prefix_block,
            story_block=story_block,
            namespace_note=self.namespace_note,
        )


# ==========================
# Validation & Combination
# ==========================

class TurtleValidator:
    @staticmethod
    def validate_ttl(ttl_content: str) -> bool:
        try:
            g = Graph()
            g.parse(data=ttl_content, format="turtle")
            return True
        except Exception as e:
            print(f"[yellow]Turtle validation failed: {e}[/yellow]")
            return False


def parse_fragment(fragment: str, prefix_block: str) -> Tuple[Graph, str]:
    """Parse an LLM fragment. If it does not parse on its own (typically because it uses
    prefixes it did not declare), retry with the prompt's prefix block prepended; the
    fragment's own declarations still win because later @prefix lines override earlier ones.
    Returns (graph, self-contained Turtle text). Raises ValueError with the parser message."""
    g = Graph()
    try:
        g.parse(data=fragment, format="turtle")
        return g, fragment
    except Exception as first:
        combined = f"{prefix_block}\n{fragment}"
        g = Graph()
        try:
            g.parse(data=combined, format="turtle")
            return g, combined
        except Exception:
            raise ValueError(f"Invalid Turtle: {first}") from first


class PropertyConstraintValidator:
    @staticmethod
    def ensure_domain_range(ttl_content: Any) -> None:
        if isinstance(ttl_content, Graph):
            g = ttl_content
        else:
            try:
                g = Graph()
                g.parse(data=ttl_content, format="turtle")
            except Exception as e:
                raise ValueError(f"Unable to parse TTL for domain/range validation: {e}") from e

        issues: List[str] = []

        def _check(prop: URIRef, prop_type: str) -> None:
            has_domain = any(True for _ in g.objects(prop, RDFS.domain))
            has_range = any(True for _ in g.objects(prop, RDFS.range))
            if not has_domain or not has_range:
                missing_parts: List[str] = []
                if not has_domain:
                    missing_parts.append("rdfs:domain")
                if not has_range:
                    missing_parts.append("rdfs:range")
                issues.append(f"{prop} ({prop_type}) missing {' and '.join(missing_parts)}")

        for prop in g.subjects(RDF.type, OWL.ObjectProperty):
            if isinstance(prop, URIRef):
                _check(prop, "ObjectProperty")
        for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
            if isinstance(prop, URIRef):
                _check(prop, "DatatypeProperty")

        if issues:
            raise ValueError("Property domain/range check failed: " + "; ".join(issues))


class OntologyCombiner:
    @staticmethod
    def remove_prefixes_from_fragment(fragment: str) -> str:
        lines = fragment.split('\n')
        cleaned: List[str] = []
        for line in lines:
            s = line.strip()
            if s.startswith('@prefix') or s.startswith('PREFIX'):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    @staticmethod
    def combine(fragments: List[Tuple[str, str]], ontology_files: List[str], prefix_block: str) -> str:
        header = (
            f"{prefix_block}\n"
            f"# Combined ontology generated from competency questions\n"
            f"# Reference ontologies: {', '.join(Path(f).name for f in ontology_files)}\n"
            f"# Generated on: {datetime.now().isoformat()}\n"
            f"# Total fragments: {len(fragments)}\n\n"
            "<http://www.example.org/ontology> a owl:Ontology .\n\n"
        )
        body: List[str] = []
        for i, (cq, fragment) in enumerate(fragments, 1):
            body.append(f"# ── Fragment {i}: {cq[:80]}{'...' if len(cq) > 80 else ''} ──\n")
            body.append(OntologyCombiner.remove_prefixes_from_fragment(fragment).strip())
            body.append("")
        return header + "\n".join(body)


# ==========================
# RAG Application
# ==========================

@dataclass
class CQOutcome:
    success: bool
    basename: str
    attempts: int = 0
    new_elements: int = 0
    changed_elements: int = 0
    error: Optional[str] = None


class OntologyRAG:
    def __init__(self, paths: Paths, normalize_vectors: bool = True, batch_size: int = 64) -> None:
        self.paths = paths
        self.tracker = TokenCostTracker()
        self.logger = RunLogger(paths, self.tracker)
        self.extractor = OntologyExtractor(self.logger)
        self.prefix_mgr = PrefixManager()
        self.embedder = get_embedder(EMBEDDER, EMBED_MODEL, self.tracker, normalize=normalize_vectors, batch_size=batch_size)
        self.store = FaissVectorStore(int(VECTOR_DIM), inner_product=True)  # cosine if normalized
        self.llm = get_llm_service(LLM_SERVICE, LLM_MODEL, LLM_TEMPERATURE, self.tracker)
        self.include_cache: bool = False
        self.reference_sources: Set[str] = set()
        self.session_sources: Set[str] = set()
        self.cache_sources: Set[str] = set()
        # incremental-update settings (set from the CLI)
        self.base_iri: Optional[str] = None
        self.max_retries: int = 0
        self.ontology_files: List[str] = []
        self.prefix_block: str = ""
        self.working: Optional[WorkingOntology] = None
        self.fragments: List[Tuple[str, str]] = []

    @staticmethod
    def _resolve_path(path: str) -> str:
        try:
            return str(Path(path).resolve())
        except Exception:
            return path

    def _allowed_sources(self) -> Set[str]:
        allowed = set(self.reference_sources)
        allowed.update(self.session_sources)
        if self.include_cache:
            allowed.update(self.cache_sources)
        return allowed

    @staticmethod
    def _sanitize_token(token: str, default: str = "item") -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "-", token.strip().lower())
        cleaned = re.sub(r"-+", "-", cleaned).strip("-")
        return cleaned or default

    def _core_tag(self, ontology_files: List[str]) -> str:
        if not ontology_files:
            return "empty"
        stems = sorted({Path(f).stem for f in ontology_files})
        combined = "-".join(stems)
        return self._sanitize_token(combined, "core")

    def _build_cq_basename(self, cq_identifier: Optional[str], seq_num: int, onto_tag: str) -> str:
        parts: List[str] = []
        if cq_identifier:
            parts.append(self._sanitize_token(cq_identifier, "id"))
        parts.append(f"cq{seq_num:03d}")
        parts.append(onto_tag)
        return "_".join(parts)

    @staticmethod
    def _extract_cq_id(row: pd.Series) -> Optional[str]:
        candidate_cols = [
            "CQID", "cqid", "CQ_ID", "cq_id", "CQId", "CQ Id",
            "ID", "Id", "id"
        ]
        for col in candidate_cols:
            if col in row and pd.notna(row[col]):
                value = str(row[col]).strip()
                if value and value.lower() != "nan":
                    return value
        return None

    @staticmethod
    def _extract_story(row: pd.Series) -> Optional[str]:
        for col in ("Story", "story", "OS", "context", "Context"):
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                return str(row[col]).strip()
        return None

    def _write_retrieved_elements(self, basename: str, scored: List[Tuple[OntologyElement, float]]) -> None:
        out_path = self.paths.retrieved_dir / f"{basename}_retrieved.csv"
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["rank", "score", "uri", "element_type", "source", "label", "domains", "ranges", "notes"])
            for rank, (el, score) in enumerate(scored, 1):
                writer.writerow([
                    rank,
                    f"{score:.6f}",
                    el.uri,
                    el.element_type,
                    Path(el.source_ontology).stem,
                    el.label or "",
                    ";".join(el.domain or []),
                    ";".join(el.range or []),
                    ";".join(el.additional_info or []),
                ])
        print(f"[blue]Retrieved elements saved to: {out_path}[/blue]")

    async def _index_elements(self, elements: List[OntologyElement]) -> None:
        new_elements = [e for e in elements if self.store.is_new(e.uri)]
        if not new_elements:
            return
        texts = [e.get_searchable_text() for e in new_elements]
        vecs = await self.embedder.embed_many(texts)
        for vec, element in zip(vecs, new_elements):
            self.store.add(vec, element)

    async def _upsert_elements(self, elements: List[OntologyElement]) -> Tuple[int, int]:
        """Insert new elements and re-embed changed ones. One element per URI (first wins,
        so an IRI that is both owl:Class and sh:NodeShape is indexed as the class)."""
        unique: Dict[str, OntologyElement] = {}
        for e in elements:
            if e.uri and e.uri not in unique:
                existing = self.store.get(e.uri)
                if existing is not None:
                    # keep the element listed under the ontology it came from
                    e = e.model_copy(update={"source_ontology": existing.source_ontology})
                unique[e.uri] = e
        if not unique:
            return 0, 0
        items = list(unique.values())
        vecs = await self.embedder.embed_many([e.get_searchable_text() for e in items])
        added = updated = 0
        for vec, element in zip(vecs, items):
            if self.store.upsert(vec, element) == "updated":
                updated += 1
            else:
                added += 1
        return added, updated

    async def _index_additional_files(self, files: List[str], *, origin: Literal["cache", "session"]) -> None:
        if not files:
            return
        elements = self.extractor.extract_from_files(files, log=False)
        await self._index_elements(elements)
        resolved = {self._resolve_path(f) for f in files}
        if origin == "cache":
            self.cache_sources.update(resolved)
        else:
            self.session_sources.update(resolved)

    async def _index_existing_fragments(self) -> None:
        if not self.include_cache:
            return
        if not self.paths.individual_dir.exists():
            return
        cache_files = sorted(self.paths.individual_dir.glob("*.ttl"))
        if not cache_files:
            return
        print(f"[yellow]Loading {len(cache_files)} cached CQ fragments into the vector store...[/yellow]")
        await self._index_additional_files([str(f) for f in cache_files], origin="cache")

    async def _build_store(self, ontology_files: List[str]) -> str:
        if ontology_files:
            print("[yellow]Building vector store from reference ontologies...[/yellow]")
        else:
            print("[yellow]No reference ontology: starting from an empty ontology[/yellow]")
        self.reference_sources = {self._resolve_path(f) for f in ontology_files}
        self.session_sources = {self._resolve_path(str(self.paths.working_ttl))}
        if not self.include_cache:
            self.cache_sources.clear()
        elements = self.extractor.extract_from_files(ontology_files) if ontology_files else []
        await self._index_elements(elements)
        await self._index_existing_fragments()
        prefix_block = self.prefix_mgr.build_prefix_block(ontology_files, base_iri=self.base_iri)
        print(f"[green]Indexed {len(self.store.elements)} ontology elements from {len(ontology_files)} ontologies[/green]")
        return prefix_block

    async def prepare(self, ontology_files: List[str]) -> None:
        """Index the starting ontology (may be empty) and set up the working graph."""
        self.ontology_files = list(ontology_files)
        self.prefix_block = await self._build_store(ontology_files)
        prefix_map = self.prefix_mgr.parse_prefix_block(self.prefix_block)
        colon = prefix_map.get("", DEFAULT_BASE_IRI)
        onto_iri = colon.rstrip("#/") or "http://example.org/ontoextend"
        self.working = WorkingOntology(prefix_map, onto_iri)
        self.working.load_base(ontology_files)
        self.fragments = []

    async def _retrieve_scored(self, cq: str, k: int = 20) -> List[Tuple[OntologyElement, float]]:
        qvec = await self.embedder.embed_one(cq)
        results = self.store.search(qvec, k)
        allowed_sources = self._allowed_sources()
        return [(el, sc) for el, sc in results if self._resolve_path(el.source_ontology) in allowed_sources]

    async def _retrieve_relevant(self, cq: str, k: int = 20) -> List[OntologyElement]:
        return [el for el, _ in await self._retrieve_scored(cq, k)]

    @staticmethod
    def _extract_ttl(response: str) -> str:
        m = re.search(r"```(?:ttl|turtle)?\s*\n(.*?)```", response, re.DOTALL)
        return (m.group(1).strip() if m else response.strip())

    async def _generate_fragment(self, cq: str, relevant: List[OntologyElement], prefix_block: str,
                                 story: Optional[str] = None) -> str:
        prompt = PromptBuilder(prefix_block, new_elements_in_base=bool(self.base_iri)).build(cq, relevant, story)
        response = await self.llm.complete(prompt, system_prompt=SYS_PROMPT)
        return self._extract_ttl(response)

    async def _generate_valid_fragment(self, cq: str, relevant: List[OntologyElement],
                                       story: Optional[str]) -> Tuple[Graph, str, int]:
        """Generate, parse and check a fragment; on failure feed the error back to the LLM up
        to `max_retries` times. Returns (graph, self-contained TTL, attempts)."""
        prompt = PromptBuilder(self.prefix_block, new_elements_in_base=bool(self.base_iri)).build(cq, relevant, story)
        response = await self.llm.complete(prompt, system_prompt=SYS_PROMPT)
        attempts = 1
        while True:
            fragment = self._extract_ttl(response)
            try:
                g, text = parse_fragment(fragment, self.prefix_block)
                PropertyConstraintValidator.ensure_domain_range(g)
                if len(g) == 0:
                    raise ValueError("Empty fragment: no triples were generated")
                return g, text, attempts
            except ValueError as err:
                if attempts > self.max_retries:
                    raise
                print(f"[yellow]Attempt {attempts} failed ({err}); asking the LLM to repair[/yellow]")
                repair = prompt + "\n\n" + REPAIR_TMPL.format(error=str(err)[:2000], fragment=fragment)
                response = await self.llm.complete(repair, system_prompt=SYS_PROMPT)
                attempts += 1

    async def process_cq(self, cq: str, basename: str, *, cq_index: Any, top_k: int = 20,
                         story: Optional[str] = None, chunk_id: Optional[str] = None) -> CQOutcome:
        """Retrieve -> generate (with optional repair) -> merge into the working ontology ->
        re-index new and changed elements. Requires `prepare()`."""
        assert self.working is not None, "call prepare() first"
        start = time.time()
        before = self.tracker.snapshot()
        scored: List[Tuple[OntologyElement, float]] = []
        src_counts: Dict[str, int] = {}
        outcome = CQOutcome(False, basename)
        try:
            scored = await self._retrieve_scored(cq, k=top_k)
            relevant = [el for el, _ in scored]
            self._write_retrieved_elements(basename, scored)
            for el in relevant:
                key = Path(el.source_ontology).stem
                src_counts[key] = src_counts.get(key, 0) + 1

            g, text, attempts = await self._generate_valid_fragment(cq, relevant, story)
            outcome.attempts = attempts
            out_file = self.paths.individual_dir / f"{basename}.ttl"
            out_file.write_text(text, encoding="utf-8")

            merge = self.working.merge(g)
            touched = merge.new_subjects | merge.changed_subjects
            full = self.working.full()
            elements = self.extractor.extract_from_graph(full, str(self.paths.working_ttl), subjects=touched)
            await self._upsert_elements(elements)
            types = {e.uri: e for e in elements}
            changes = [(str(u), "new", types[str(u)].element_type if str(u) in types else "", types[str(u)].label or "" if str(u) in types else "")
                       for u in sorted(merge.new_subjects)]
            changes += [(str(u), "changed", types[str(u)].element_type if str(u) in types else "", types[str(u)].label or "" if str(u) in types else "")
                        for u in sorted(merge.changed_subjects)]
            self.logger.log_provenance(cq_index, basename, cq, chunk_id, changes)
            self.logger.log_duplicates(cq_index, basename, merge.duplicate_candidates)
            self.paths.working_ttl.write_text(full.serialize(format="turtle"), encoding="utf-8")
            self.fragments.append((cq, text))

            outcome.success = True
            outcome.new_elements = len(merge.new_subjects)
            outcome.changed_elements = len(merge.changed_subjects)
            print(f"[green]{basename}: merged {merge.added_triples} triples "
                  f"({outcome.new_elements} new, {outcome.changed_elements} changed elements)[/green]")
        except ValueError as err:
            outcome.error = str(err)
            outcome.attempts = outcome.attempts or (self.max_retries + 1)
            print(f"[red]{basename}: {err}[/red]")
        except Exception as e:
            outcome.error = f"{type(e).__name__}: {e}"
            print(f"[red]{basename} failed: {outcome.error}[/red]")
        finally:
            self.logger.log_processing_result(
                cq_index, cq, outcome.success, len(scored), src_counts, time.time() - start,
                outcome.error, attempts=outcome.attempts, new_elements=outcome.new_elements,
                changed_elements=outcome.changed_elements, chunk_id=chunk_id,
            )
            self.logger.log_cost_row(cq_index, cq, self.tracker.diff(before, self.tracker.snapshot()))
        return outcome

    def finalize(self, tag: str) -> Optional[Path]:
        """Write <tag>_combined.ttl (extension only) and <tag>_updated.ttl (start + extension)."""
        assert self.working is not None
        if not self.fragments:
            return None
        combined_path = self.paths.cache_dir / f"{tag}_combined.ttl"
        updated_path = self.paths.cache_dir / f"{tag}_updated.ttl"
        self.working.serialize_extension(combined_path, self.ontology_files, len(self.fragments))
        self.working.serialize_updated(updated_path)
        for pth in (combined_path, updated_path):
            ok = TurtleValidator.validate_ttl(pth.read_text(encoding="utf-8"))
            print(f"[green]Saved {pth} ({'valid' if ok else 'INVALID'} Turtle)[/green]")
        return combined_path

    async def run_single_cq(self, cq: str, ontology_files: List[str], top_k: int = 20, story: Optional[str] = None) -> None:
        await self.prepare(ontology_files)
        onto_tag = self._core_tag(ontology_files)
        basename = self._build_cq_basename(None, 1, onto_tag)
        outcome = await self.process_cq(cq, basename, cq_index=-1, top_k=top_k, story=story)
        if outcome.success:
            print("\n[bold green]Ontology fragment[/bold green]\n")
            print(self.fragments[-1][1])
            self.finalize(f"single_{onto_tag}")
        print(f"[bold green]💲 Estimated total API cost (USD): {self.tracker.totals_cost_usd():.6f}[/bold green]")

    async def run_csv(self, csv_file: str, ontology_files: List[str], limit: Optional[int] = None, top_k: int = 20) -> None:
        print(f"[bold blue]Processing CQ dataset: {csv_file}[/bold blue]")
        print(f"[bold blue]Using {len(ontology_files)} reference ontologies:[/bold blue]")
        for i, onto_file in enumerate(ontology_files, 1):
            print(f"[cyan]  {i}. {Path(onto_file).name}[/cyan]")

        if not Path(csv_file).exists():
            print(f"[red]CSV file not found: {csv_file}[/red]")
            return
        missing = [f for f in ontology_files if not Path(f).exists()]
        if missing:
            print(f"[red]Ontology files not found: {missing}[/red]")
            return

        try:
            df = pd.read_csv(csv_file)
            print(f"[green]Loaded {len(df)} rows from CSV[/green]")
        except Exception as e:
            print(f"[red]Error reading CSV: {e}[/red]")
            return

        if 'CQ' not in df.columns:
            print(f"[red]'CQ' column not found in CSV. Available columns: {list(df.columns)}[/red]")
            return

        if limit:
            df = df.head(limit)
            print(f"[blue]Processing limited to first {len(df)} rows[/blue]")

        await self.prepare(ontology_files)
        onto_tag = self._core_tag(ontology_files)

        successful = 0
        failed = 0
        start_time = time.time()

        from tqdm import tqdm  # local import to avoid mandatory dep during static analysis
        pbar = tqdm(total=len(df), desc="Processing CQs")

        for seq_num, (idx, row) in enumerate(df.iterrows(), start=1):
            cq = str(row['CQ']).strip()
            if not cq or cq.lower() in ['nan', 'none', '']:
                print(f"[yellow]Row {idx}: Empty CQ, skipping[/yellow]")
                self.logger.log_processing_result(idx, cq, False, 0, {}, 0, "Empty CQ")
                failed += 1
                pbar.update(1)
                continue
            pbar.set_description(f"Processing CQ {idx}")
            basename = self._build_cq_basename(self._extract_cq_id(row), seq_num, onto_tag)
            chunk_id = str(row["chunk_id"]) if "chunk_id" in row and pd.notna(row["chunk_id"]) else None
            outcome = await self.process_cq(cq, basename, cq_index=idx, top_k=top_k,
                                            story=self._extract_story(row), chunk_id=chunk_id)
            successful += int(outcome.success)
            failed += int(not outcome.success)
            pbar.update(1)
            await asyncio.sleep(0.1)

        pbar.close()

        combined_path = self.finalize(f"{Path(csv_file).stem}_{onto_tag}") or self.paths.combined_ttl
        total_time = time.time() - start_time
        self.logger.log_final_results(len(df), successful, failed, total_time, str(combined_path))

        print(f"\n[bold green]Processing Complete![/bold green]")
        print(f"Total CQs: {len(df)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        if len(df):
            print(f"Success rate: {(successful / len(df) * 100):.1f}%")
        print(f"[bold green]💲 Estimated total API cost (USD): {self.tracker.totals_cost_usd():.6f}[/bold green]")


    async def run_document(self, doc_file: str, ontology_files: List[str], *, top_k: int = 20,
                           max_tokens: int = 400, overlap_tokens: int = 0, rechunk: bool = False,
                           max_cqs: int = 5, dedup_threshold: float = 0.92, require_evidence: bool = True,
                           use_story: bool = True, cq_only: bool = False, limit_chunks: Optional[int] = None) -> None:
        """Parsed document -> chunks -> CQs per chunk -> incremental ontology extension.

        Chunks are processed in document order; each chunk's CQs are generated and then
        immediately used to extend the working ontology, so later chunks can reuse what
        earlier chunks introduced."""
        def count(text: str) -> int:
            try:
                return len(ENC.encode(text))
            except Exception:
                return doc_pipeline.approx_tokens(text)

        chunks = doc_pipeline.load_chunks(doc_file, max_tokens=max_tokens, overlap_tokens=overlap_tokens,
                                          rechunk=rechunk, count=count)
        if limit_chunks:
            chunks = chunks[:limit_chunks]
        doc_pipeline.write_chunks_jsonl(chunks, self.paths.cache_dir / "chunks.jsonl")
        print(f"[bold blue]Document {doc_file}: {len(chunks)} chunks[/bold blue]")

        if not cq_only:
            await self.prepare(ontology_files)
        gen = doc_pipeline.CQGenerator(self.llm, self.embedder, max_cqs=max_cqs, dedup_threshold=dedup_threshold,
                                       require_evidence=require_evidence, doc_title=Path(doc_file).stem)
        onto_tag = self._core_tag(ontology_files)
        cq_csv = self.paths.cache_dir / "generated_cqs.csv"
        header = ["CQID", "CQ", "chunk_id", "chunk_order", "heading", "evidence", "type", "status",
                  "duplicate_of", "similarity", "extension_success", "new_elements", "changed_elements", "Story"]
        with cq_csv.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(header)

        stats = {"chunks": len(chunks), "candidates": 0, "kept": 0, "duplicate": 0, "ungrounded": 0,
                 "extended_ok": 0, "extended_failed": 0}
        start_time = time.time()
        seq = 0
        for chunk in chunks:
            before = self.tracker.snapshot()
            try:
                candidates = await gen.generate(chunk)
            except Exception as e:
                print(f"[red]Chunk {chunk.chunk_id}: CQ generation failed: {e}[/red]")
                candidates = []
            self.logger.log_cost_row(f"{chunk.chunk_id}:cqgen", chunk.text[:200], self.tracker.diff(before, self.tracker.snapshot()))
            story = chunk.context_text() if use_story else None
            print(f"[cyan]Chunk {chunk.chunk_id} ({' > '.join(chunk.heading_path)}): "
                  f"{sum(c.status == 'kept' for c in candidates)}/{len(candidates)} CQs kept[/cyan]")
            for c in candidates:
                stats["candidates"] += 1
                stats[c.status] = stats.get(c.status, 0) + 1
                ok, new_n, chg_n = "", "", ""
                if c.status == "kept" and not cq_only:
                    seq += 1
                    basename = self._build_cq_basename(c.cq_id, seq, onto_tag)
                    outcome = await self.process_cq(c.cq, basename, cq_index=c.cq_id, top_k=top_k,
                                                    story=story, chunk_id=chunk.chunk_id)
                    ok, new_n, chg_n = outcome.success, outcome.new_elements, outcome.changed_elements
                    stats["extended_ok" if outcome.success else "extended_failed"] += 1
                with cq_csv.open("a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow([
                        c.cq_id, c.cq, c.chunk_id, chunk.order, " > ".join(chunk.heading_path), c.evidence,
                        c.cq_type, c.status, c.duplicate_of, f"{c.similarity:.4f}", ok, new_n, chg_n,
                        chunk.context_text(),
                    ])

        combined_path: Optional[Path] = None
        if not cq_only:
            combined_path = self.finalize(f"{self._sanitize_token(Path(doc_file).stem, 'doc')}_{onto_tag}")
            self.logger.log_final_results(stats["kept"], stats["extended_ok"], stats["extended_failed"],
                                          time.time() - start_time, str(combined_path or ""))
        (self.paths.cache_dir / "document_run_summary.json").write_text(
            json.dumps({**stats, "document": doc_file, "reference_ontologies": ontology_files,
                                      "base_iri": self.base_iri, "output": str(combined_path or "")}, indent=2),
            encoding="utf-8")
        print(f"\n[bold green]Document processing complete[/bold green]: {stats}")
        print(f"CQs written to {cq_csv}")
        print(f"[bold green]💲 Estimated total API cost (USD): {self.tracker.totals_cost_usd():.6f}[/bold green]")


# ==========================
# CLI
# ==========================


def expand_onto_args(files: Optional[List[str]], dirs: Optional[List[str]]) -> List[str]:
    candidates: List[str] = []
    if files:
        candidates.extend(files)
    if dirs:
        for d in dirs:
            p = Path(d)
            if not p.exists() or not p.is_dir():
                print(f"[yellow]Not a directory (skipped): {d}[/yellow]")
                continue
            candidates.extend(str(f) for f in p.rglob("*.ttl"))
            candidates.extend(str(f) for f in p.rglob("*.owl"))
    seen = set()
    result: List[str] = []
    for f in candidates:
        if f not in seen:
            result.append(f)
            seen.add(f)
    if not result:
        if files or dirs:
            print("[red]No ontology files found from --onto/--onto-dir[/red]")
            sys.exit(1)
        print("[yellow]No --onto/--onto-dir given: starting from an empty ontology[/yellow]")
    return result


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--onto", nargs="+", help="Reference/base ontology files (.ttl/.owl). Omit both --onto and --onto-dir to start from an empty ontology.")
    sp.add_argument("--onto-dir", nargs="+", help="One or more directories to scan for .ttl/.owl")
    sp.add_argument("--top-k", type=int, default=20, help="Top-K elements to retrieve")
    sp.add_argument("--include-cache", action="store_true", help="Also reuse previously generated CQ fragments from cache")
    sp.add_argument("--run-dir", default=None, help="Directory for logs, fragments and combined output (default: ~/.core_ontology_rag_cache). Use a fresh one per experiment run.")
    sp.add_argument("--base-iri", default=None, help=f"Namespace bound to ':' for NEW elements (prompt asks the LLM to put new elements there). Default: the reference ontology's default namespace, or {DEFAULT_BASE_IRI} when there is none.")
    sp.add_argument("--max-retries", type=int, default=0, help="Self-correction: re-prompt the LLM with the validation error up to N times (default 0 = original behaviour)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OntoExtend: CQ / document -> ontology extension via RAG")
    sub = p.add_subparsers(dest="cmd")

    p_doc = sub.add_parser("doc", help="Parsed document -> chunks -> CQs per chunk -> ontology extension")
    p_doc.add_argument("--file", required=True, help="Parsed document (.md, .txt, .json, .jsonl)")
    _add_common(p_doc)
    p_doc.add_argument("--chunk-tokens", type=int, default=400, help="Max tokens per chunk (default 400)")
    p_doc.add_argument("--chunk-overlap", type=int, default=0, help="Tokens of trailing paragraphs repeated in the next chunk")
    p_doc.add_argument("--rechunk", action="store_true", help="Ignore chunk boundaries in JSON input and re-chunk the concatenated text")
    p_doc.add_argument("--max-cqs", type=int, default=5, help="Max CQs per chunk (default 5)")
    p_doc.add_argument("--dedup-threshold", type=float, default=0.92, help="Cosine similarity above which a CQ counts as a duplicate of an earlier one")
    p_doc.add_argument("--keep-ungrounded", action="store_true", help="Keep CQs whose evidence quote is not found in the chunk")
    p_doc.add_argument("--no-story", action="store_true", help="Do not pass the chunk text to the extension prompt as ONTOLOGY STORY")
    p_doc.add_argument("--cq-only", action="store_true", help="Only chunk and generate CQs (writes generated_cqs.csv), no extension")
    p_doc.add_argument("--limit-chunks", type=int, default=None, help="Process only the first N chunks")

    p_single = sub.add_parser("cq", help="Generate ontology for a single CQ")
    p_single.add_argument("--cq", required=True, help="Competency Question text")
    p_single.add_argument("--story", default=None, help="Optional ontology story / source text for the CQ")
    _add_common(p_single)

    p_batch = sub.add_parser("csv", help="Process a CSV with a CQ column")
    p_batch.add_argument("--file", required=True, help="CSV file with column 'CQ' (optional: CQID, Story, chunk_id)")
    p_batch.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    _add_common(p_batch)

    return p.parse_args()


async def _main_async(args: argparse.Namespace) -> None:
    run_dir = getattr(args, "run_dir", None)
    paths = Paths(Path(run_dir).expanduser()) if run_dir else Paths()
    if run_dir:
        print(f"[blue]Writing all outputs to run directory: {paths.cache_dir}[/blue]")
    app = OntologyRAG(paths)

    if not args.cmd:
        print("[bold red]Usage:[/bold red] python rag-ontoextend.py cq --cq \"...\" [--onto core.ttl ...]")
        print("         python rag-ontoextend.py csv --file dataset.csv [--onto core.ttl ...] [--limit 50]")
        print("         python rag-ontoextend.py doc --file parsed_doc.md [--onto-dir base/] [--base-iri IRI]")
        sys.exit(1)

    app.include_cache = bool(getattr(args, "include_cache", False))
    app.base_iri = args.base_iri
    app.max_retries = max(0, args.max_retries)
    ontofiles = expand_onto_args(args.onto, args.onto_dir)
    if args.cmd == "cq":
        await app.run_single_cq(args.cq, ontofiles, top_k=args.top_k, story=args.story)
    elif args.cmd == "csv":
        await app.run_csv(args.file, ontofiles, limit=args.limit, top_k=args.top_k)
    elif args.cmd == "doc":
        await app.run_document(
            args.file, ontofiles, top_k=args.top_k, max_tokens=args.chunk_tokens,
            overlap_tokens=args.chunk_overlap, rechunk=args.rechunk, max_cqs=args.max_cqs,
            dedup_threshold=args.dedup_threshold, require_evidence=not args.keep_ungrounded,
            use_story=not args.no_story, cq_only=args.cq_only, limit_chunks=args.limit_chunks,
        )


def main() -> None:
    args = parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()


#python filename.py csv --file .\cqs.csv --onto-dir .\coreontologies --limit 3
