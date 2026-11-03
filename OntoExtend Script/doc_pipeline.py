"""Document -> chunks -> competency questions, the front end of document-based ontology extension.

The input is an already *parsed* document (the parser itself is out of scope). Supported shapes:

* ``.md`` / ``.markdown`` / ``.txt``: plain text; Markdown headings (``#``...) are used as section
  boundaries and kept as the chunk's heading path.
* ``.json``: a list of chunks, or an object with a ``chunks`` / ``sections`` / ``pages`` /
  ``elements`` / ``blocks`` list. Each item is a string or an object with a text field
  (``text``, ``content``, ``markdown``, ``md`` or ``body``) and optional ``id`` / ``heading`` /
  ``title`` / ``section`` / ``page`` fields.
* ``.jsonl``: one such object per line.

Pre-chunked JSON is kept as-is unless ``rechunk=True``; long items are still split so a chunk never
exceeds ``max_tokens`` by much.

CQ generation asks the LLM for a small JSON list of CQs per chunk, each with a verbatim evidence
quote. CQs whose evidence is not found in the chunk are dropped (grounding check), and near-duplicate
CQs across chunks are dropped by normalized text and embedding similarity.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

TokenCounter = Callable[[str], int]


def approx_tokens(text: str) -> int:
    """Cheap fallback when no tokenizer is supplied (~0.75 words per token)."""
    return int(len(re.findall(r"\S+", text)) * 1.33) + 1


# ==========================
# Chunks
# ==========================

@dataclass
class Chunk:
    chunk_id: str
    order: int
    text: str
    heading_path: List[str] = field(default_factory=list)
    source: str = ""
    page: Optional[str] = None

    def context_text(self, max_chars: int = 4000) -> str:
        head = " > ".join(self.heading_path)
        body = self.text if len(self.text) <= max_chars else self.text[:max_chars] + " ..."
        return f"[Section: {head}]\n{body}" if head else body


_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_TEXT_KEYS = ("text", "content", "markdown", "md", "body")
_LIST_KEYS = ("chunks", "sections", "pages", "elements", "blocks", "items")


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_long(paragraph: str, max_tokens: int, count: TokenCounter) -> List[str]:
    if count(paragraph) <= max_tokens:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?。])\s+", paragraph)
    out: List[str] = []
    cur = ""
    for s in sentences:
        cand = f"{cur} {s}".strip()
        if cur and count(cand) > max_tokens:
            out.append(cur)
            cur = s
        else:
            cur = cand
    if cur:
        out.append(cur)
    # a single sentence longer than max_tokens: hard split on words
    final: List[str] = []
    for piece in out:
        if count(piece) <= max_tokens * 1.5:
            final.append(piece)
            continue
        words = piece.split()
        step = max(1, int(max_tokens / 1.33))
        final.extend(" ".join(words[i:i + step]) for i in range(0, len(words), step))
    return final


def _pack(paragraphs: List[str], max_tokens: int, overlap_tokens: int, count: TokenCounter) -> List[str]:
    """Greedy packing of paragraphs into chunks; trailing paragraphs up to `overlap_tokens`
    are repeated at the start of the next chunk."""
    pieces: List[str] = []
    for p in paragraphs:
        pieces.extend(_split_long(p, max_tokens, count))
    chunks: List[str] = []
    cur: List[str] = []
    for p in pieces:
        if cur and count("\n\n".join(cur + [p])) > max_tokens:
            chunks.append("\n\n".join(cur))
            carry: List[str] = []
            if overlap_tokens > 0:
                for q in reversed(cur):
                    if count("\n\n".join([q] + carry)) > overlap_tokens:
                        break
                    carry.insert(0, q)
            cur = carry
        cur.append(p)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_markdown(text: str, *, source: str = "", max_tokens: int = 400, overlap_tokens: int = 0,
                   count: TokenCounter = approx_tokens, id_prefix: str = "c") -> List[Chunk]:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # parser/editor comments are not content
    sections: List[Tuple[List[str], List[str]]] = []  # (heading_path, lines)
    path: List[str] = []
    lines: List[str] = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else _HEADING.match(line)
        if m:
            if any(l.strip() for l in lines):
                sections.append((list(path), lines))
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            lines = []
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        sections.append((list(path), lines))

    out: List[Chunk] = []
    for heading_path, sec_lines in sections:
        for body in _pack(_split_paragraphs("\n".join(sec_lines)), max_tokens, overlap_tokens, count):
            out.append(Chunk(f"{id_prefix}{len(out) + 1:04d}", len(out), body, heading_path, source))
    return out


def _item_text(item: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(item, str):
        return item, {}
    if isinstance(item, dict):
        for k in _TEXT_KEYS:
            if isinstance(item.get(k), str):
                return item[k], item
    return "", {}


def _json_items(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in _LIST_KEYS:
            if isinstance(data.get(k), list):
                return data[k]
        if any(isinstance(data.get(k), str) for k in _TEXT_KEYS):
            return [data]
    raise ValueError("Unrecognised JSON document: expected a list of chunks or an object with a "
                     f"{'/'.join(_LIST_KEYS)} list")


def load_chunks(path: str, *, max_tokens: int = 400, overlap_tokens: int = 0, rechunk: bool = False,
                count: TokenCounter = approx_tokens) -> List[Chunk]:
    p = Path(path)
    suffix = p.suffix.lower()
    raw = p.read_text(encoding="utf-8")
    if suffix in (".md", ".markdown", ".txt", ""):
        return chunk_markdown(raw, source=p.name, max_tokens=max_tokens, overlap_tokens=overlap_tokens, count=count)

    if suffix == ".jsonl":
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    elif suffix == ".json":
        items = _json_items(json.loads(raw))
    else:
        raise ValueError(f"Unsupported document type: {suffix} (use .md, .txt, .json or .jsonl)")

    if rechunk:
        text = "\n\n".join(_item_text(it)[0] for it in items)
        return chunk_markdown(text, source=p.name, max_tokens=max_tokens, overlap_tokens=overlap_tokens, count=count)

    out: List[Chunk] = []
    for it in items:
        text, meta = _item_text(it)
        if not text.strip():
            continue
        heading = meta.get("heading_path") or meta.get("heading") or meta.get("title") or meta.get("section")
        heading_path = heading if isinstance(heading, list) else ([str(heading)] if heading else [])
        page = meta.get("page") or meta.get("page_number")
        base_id = str(meta.get("id") or meta.get("chunk_id") or f"c{len(out) + 1:04d}")
        parts = _pack(_split_paragraphs(text) or [text], max_tokens, overlap_tokens, count) \
            if count(text) > max_tokens * 1.5 else [text.strip()]
        for j, part in enumerate(parts):
            cid = base_id if len(parts) == 1 else f"{base_id}-{j + 1}"
            out.append(Chunk(cid, len(out), part, [str(h) for h in heading_path], p.name,
                             str(page) if page is not None else None))
    return out


def write_chunks_jsonl(chunks: Iterable[Chunk], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


# ==========================
# CQ generation
# ==========================

CQ_SYS_PROMPT = (
    "You are an ontology engineer who writes competency questions (CQs) for ontology requirements. "
    "Return only JSON."
)

CQ_GEN_TMPL = """Read the TEXT CHUNK below (taken from a larger document{doc_title}) and write competency questions (CQs) that an ontology should be able to answer so that it captures the domain knowledge stated in this chunk.

Rules:
1. Write at most {max_cqs} CQs. Fewer is better than vague or redundant ones. If the chunk contains no domain knowledge (e.g. a table of contents, references, legal boilerplate), return an empty list.
2. Each CQ must be answerable from the knowledge stated in the chunk; do not use outside knowledge.
3. Ask about kinds of things, their properties and their relations (the schema), not about one specific instance. Use X for specific values or identifiers, e.g. "Which machine executed the process with id X?".
4. Use the domain terms exactly as the chunk uses them, so they can be matched to ontology elements.
5. Each CQ must be one self-contained question; do not refer to "the text" or "this section".
6. For each CQ copy a short VERBATIM quote (one sentence or less) from the chunk as evidence.
7. The CQs must be in English, even if the chunk is written in another language; keep the evidence quote in the chunk's original language.
{existing_block}
Return JSON only, in this form:
{{"cqs": [{{"cq": "...", "evidence": "...", "type": "class|relation|attribute|constraint"}}]}}

TEXT CHUNK{heading}:
\"\"\"
{chunk}
\"\"\"
"""


@dataclass
class GeneratedCQ:
    cq_id: str
    chunk_id: str
    cq: str
    evidence: str
    cq_type: str
    grounded: bool
    status: str = "kept"         # kept | duplicate | ungrounded
    duplicate_of: str = ""
    similarity: float = 0.0


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _norm_quote(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return re.sub(r"[^\w]+", " ", _norm_ws(text)).strip()


def is_grounded(evidence: str, chunk_text: str) -> bool:
    ev = _norm_quote(evidence)
    if not ev:
        return False
    body = _norm_quote(chunk_text)
    if ev in body:
        return True
    # tolerate small edits: most of the evidence's 4-word shingles must occur in the chunk
    words = ev.split()
    if len(words) < 6:
        return False
    shingles = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    return sum(s in body for s in shingles) / len(shingles) >= 0.8


def parse_cq_json(response: str) -> List[Dict[str, str]]:
    text = response.strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    items = data.get("cqs", []) if isinstance(data, dict) else data
    out: List[Dict[str, str]] = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, str):
            out.append({"cq": it, "evidence": "", "type": ""})
        elif isinstance(it, dict) and str(it.get("cq", "")).strip():
            out.append({"cq": str(it["cq"]).strip(), "evidence": str(it.get("evidence", "")).strip(),
                        "type": str(it.get("type", "")).strip()})
    return out


class CQGenerator:
    """Generates CQs per chunk and filters them (grounding + cross-chunk de-duplication)."""

    def __init__(self, llm: Any, embedder: Any, *, max_cqs: int = 5, dedup_threshold: float = 0.92,
                 require_evidence: bool = True, doc_title: str = "", show_previous: int = 0) -> None:
        self.llm = llm
        self.embedder = embedder
        self.max_cqs = max_cqs
        self.dedup_threshold = dedup_threshold
        self.require_evidence = require_evidence
        self.doc_title = doc_title
        self.show_previous = show_previous
        self.kept: List[GeneratedCQ] = []
        self._kept_vecs: List[np.ndarray] = []
        self._kept_norm: Dict[str, str] = {}
        self._counter = 0

    def build_prompt(self, chunk: Chunk) -> str:
        existing_block = ""
        if self.show_previous and self.kept:
            prev = "\n".join(f"- {c.cq}" for c in self.kept[-self.show_previous:])
            existing_block = f"8. These CQs already exist for earlier chunks; do not repeat them:\n{prev}\n"
        heading = f" (section: {' > '.join(chunk.heading_path)})" if chunk.heading_path else ""
        return CQ_GEN_TMPL.format(
            doc_title=f' "{self.doc_title}"' if self.doc_title else "",
            max_cqs=self.max_cqs,
            existing_block=existing_block,
            heading=heading,
            chunk=chunk.text,
        )

    async def generate(self, chunk: Chunk) -> List[GeneratedCQ]:
        """Returns every candidate for the chunk (with status); only status=='kept' should be used."""
        response = await self.llm.complete(self.build_prompt(chunk), system_prompt=CQ_SYS_PROMPT)
        raw = parse_cq_json(response)[: self.max_cqs]
        out: List[GeneratedCQ] = []
        for item in raw:
            self._counter += 1
            cq = item["cq"].rstrip()
            if not cq.endswith("?"):
                cq += "?"
            g = GeneratedCQ(f"{chunk.chunk_id}-q{len(out) + 1}", chunk.chunk_id, cq, item["evidence"],
                            item["type"], is_grounded(item["evidence"], chunk.text))
            if self.require_evidence and not g.grounded:
                g.status = "ungrounded"
                out.append(g)
                continue
            norm = _norm_quote(cq)
            if norm in self._kept_norm:
                g.status, g.duplicate_of, g.similarity = "duplicate", self._kept_norm[norm], 1.0
                out.append(g)
                continue
            vec = np.asarray(await self.embedder.embed_one(cq), dtype="float32").reshape(-1)
            n = np.linalg.norm(vec)
            vec = vec / n if n else vec
            if self._kept_vecs:
                sims = np.stack(self._kept_vecs) @ vec
                j = int(np.argmax(sims))
                if float(sims[j]) >= self.dedup_threshold:
                    g.status, g.duplicate_of, g.similarity = "duplicate", self.kept[j].cq_id, float(sims[j])
                    out.append(g)
                    continue
            self.kept.append(g)
            self._kept_vecs.append(vec)
            self._kept_norm[norm] = g.cq_id
            out.append(g)
        return out
