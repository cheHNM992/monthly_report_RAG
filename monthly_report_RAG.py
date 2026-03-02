#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

LOGGER = logging.getLogger("monthly_report_RAG")
EXCLUDED_TAGS = {"script", "style", "noscript"}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_GPT_MODEL = "gpt-5.2-2025-12-11"


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: List[str] = []
        self._in_h1 = False
        self._h1_parts: List[str] = []
        self._current_heading_stack: List[str] = []
        self._block_buffer: List[str] = []
        self.blocks: List[Tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in EXCLUDED_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return

        if tag == "title":
            self._in_title = True
        elif tag == "h1":
            self._in_h1 = True

        if tag in {"p", "div", "li", "br", "tr"}:
            self._flush_block()
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_block()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in EXCLUDED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return

        if tag == "title":
            self._in_title = False
        elif tag == "h1":
            self._in_h1 = False

        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_block()

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        normalized = " ".join(data.split())
        if not normalized:
            return

        if self._in_title:
            self._title_parts.append(normalized)

        if self._in_h1:
            self._h1_parts.append(normalized)

        self._block_buffer.append(normalized)

    def _flush_block(self) -> None:
        if not self._block_buffer:
            return
        block_text = " ".join(self._block_buffer).strip()
        self._block_buffer.clear()
        if not block_text:
            return

        if any(block_text.startswith(prefix) for prefix in ("#", "＃")):
            self._current_heading_stack = [block_text]
        self.blocks.append((" > ".join(self._current_heading_stack), block_text))

    def close(self) -> None:
        self._flush_block()
        super().close()

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()

    @property
    def h1(self) -> str:
        return " ".join(self._h1_parts).strip()


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_html_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".html", ".htm"}:
            yield p


def extract_html(path: Path) -> Tuple[str, List[Tuple[str, str]], str]:
    try:
        parser = HTMLTextExtractor()
        parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
        parser.close()
        full_text = normalize_text("\n".join(block for _, block in parser.blocks))
        title = parser.title or parser.h1 or path.stem
        return title, parser.blocks, full_text
    except Exception as exc:
        raise RuntimeError(f"HTML parse failed: {path}: {exc}")


def split_chunks(blocks: List[Tuple[str, str]], chunk_size: int = 500) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    current_heading = ""
    current_parts: List[str] = []
    current_len = 0
    min_chunk = 300
    max_chunk = 800

    def flush() -> None:
        nonlocal current_parts, current_len, current_heading
        if not current_parts:
            return
        text = normalize_text("\n".join(current_parts))
        if text:
            chunks.append({"heading_path": current_heading, "text": text})
        current_parts = []
        current_len = 0

    target = min(max(chunk_size, min_chunk), max_chunk)

    for heading, block in blocks:
        block = normalize_text(block)
        if not block:
            continue
        if len(block) > max_chunk:
            flush()
            start = 0
            while start < len(block):
                part = block[start : start + target]
                chunks.append({"heading_path": heading, "text": part})
                start += target
            continue

        if current_len + len(block) + 1 > target and current_len >= min_chunk:
            flush()

        if not current_parts:
            current_heading = heading
        current_parts.append(block)
        current_len += len(block) + 1

    flush()
    return chunks


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def with_retry(func, *args, retries: int = 3, **kwargs):
    last_err = None
    for i in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_err = exc
            if i < retries:
                sleep_s = 1.5 * i
                LOGGER.warning("API call failed (%s/%s): %s. retrying in %.1fs", i, retries, exc, sleep_s)
                time.sleep(sleep_s)
    raise RuntimeError(f"API call failed after {retries} retries: {last_err}")


class OpenAIClient:
    def __init__(self, embedding_model: str, chat_model: str):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is required. Install with `pip install openai`.") from exc

        self._client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        def _call():
            res = self._client.embeddings.create(model=self.embedding_model, input=texts)
            return [d.embedding for d in res.data]

        return with_retry(_call)

    def generate_answer(self, prompt: str, max_tokens: int) -> str:
        def _call():
            res = self._client.responses.create(
                model=self.chat_model,
                input=prompt,
                max_output_tokens=max_tokens,
            )
            return res.output_text

        return with_retry(_call)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_index(root: Path, index_dir: Path, chunk_size: int, embedding_model: str, gpt_model: str) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    docs_path = index_dir / "documents.json"
    chunks_path = index_dir / "chunks.json"

    old_docs: List[Dict[str, Any]] = load_json(docs_path, [])
    old_chunks: List[Dict[str, Any]] = load_json(chunks_path, [])
    old_docs_by_path = {d["file_path"]: d for d in old_docs}

    new_docs: List[Dict[str, Any]] = []
    new_chunks: List[Dict[str, Any]] = [c for c in old_chunks]

    client = OpenAIClient(embedding_model=embedding_model, chat_model=gpt_model)

    current_files = sorted(iter_html_files(root))
    current_paths = {str(p.resolve()) for p in current_files}

    # Remove deleted docs/chunks
    deleted_doc_ids = {d["doc_id"] for d in old_docs if d["file_path"] not in current_paths}
    if deleted_doc_ids:
        new_chunks = [c for c in new_chunks if c["doc_id"] not in deleted_doc_ids]

    for i, fpath in enumerate(current_files, start=1):
        abs_path = str(fpath.resolve())
        stat = fpath.stat()
        file_hash = file_sha256(fpath)
        existing = old_docs_by_path.get(abs_path)

        unchanged = (
            existing
            and float(existing.get("mtime", -1)) == float(stat.st_mtime)
            and existing.get("hash") == file_hash
        )
        if unchanged:
            new_docs.append(existing)
            continue

        doc_id = existing["doc_id"] if existing else hashlib.md5(abs_path.encode()).hexdigest()
        new_chunks = [c for c in new_chunks if c["doc_id"] != doc_id]

        try:
            title, blocks, _full_text = extract_html(fpath)
        except Exception as exc:
            LOGGER.warning("Skipping file due to parse error: %s", exc)
            continue

        chunk_items = split_chunks(blocks, chunk_size=chunk_size)
        texts = [c["text"] for c in chunk_items]

        embeddings: List[List[float]] = []
        if texts:
            try:
                embeddings = client.embed_texts(texts)
            except Exception as exc:
                LOGGER.error("Embedding failed for doc %s: %s", abs_path, exc)
                embeddings = [None for _ in texts]  # type: ignore

        for idx, c in enumerate(chunk_items):
            emb = embeddings[idx] if idx < len(embeddings) else None
            if emb is None:
                LOGGER.error("Embedding missing for chunk %s#%s", abs_path, idx)
                continue
            new_chunks.append(
                {
                    "chunk_id": f"{doc_id}:{idx}",
                    "doc_id": doc_id,
                    "file_path": abs_path,
                    "title": title,
                    "heading_path": c["heading_path"],
                    "chunk_index": idx,
                    "text": c["text"],
                    "embedding_vector": emb,
                }
            )

        new_docs.append(
            {
                "doc_id": doc_id,
                "file_path": abs_path,
                "title": title,
                "mtime": stat.st_mtime,
                "hash": file_hash,
            }
        )
        LOGGER.info("Indexed %s (%s/%s)", abs_path, i, len(current_files))

    save_json(docs_path, new_docs)
    save_json(chunks_path, new_chunks)
    LOGGER.info("Index complete. docs=%s chunks=%s", len(new_docs), len(new_chunks))


def query_index(
    index_dir: Path,
    query: str,
    top_k: int,
    limit_docs: int,
    embedding_model: str,
    gpt_model: str,
    max_tokens: int,
) -> None:
    docs = load_json(index_dir / "documents.json", [])
    chunks = load_json(index_dir / "chunks.json", [])
    if not docs or not chunks:
        raise RuntimeError("Index is empty. Run `index` first.")

    client = OpenAIClient(embedding_model=embedding_model, chat_model=gpt_model)
    q_embedding = client.embed_texts([query])[0]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for c in chunks:
        sim = cosine_similarity(q_embedding, c["embedding_vector"])
        scored.append((sim, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:top_k]

    doc_scores: Dict[str, float] = {}
    doc_best_chunks: Dict[str, List[Dict[str, Any]]] = {}
    for score, chunk in top_chunks:
        doc_id = chunk["doc_id"]
        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
        doc_best_chunks.setdefault(doc_id, []).append(chunk)

    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:limit_docs]

    references: List[Dict[str, str]] = []
    for doc_id, _score in ranked_docs:
        best = doc_best_chunks[doc_id][:2]
        for chunk in best:
            author, year_month = infer_author_and_year_month(
                file_path=chunk.get("file_path", ""),
                title=chunk.get("title", ""),
                excerpt=chunk.get("text", ""),
            )
            references.append(
                {
                    "title": chunk.get("title", ""),
                    "file_path": chunk.get("file_path", ""),
                    "excerpt": chunk.get("text", ""),
                    "author": author,
                    "year_month": year_month,
                }
            )

    max_chars = max_tokens * 4
    ref_texts: List[str] = []
    used = 0
    for i, ref in enumerate(references, start=1):
        block = (
            f"[{i}] title: {ref['title']}\n"
            f"path: {ref['file_path']}\n"
            f"author: {ref['author']}\n"
            f"year_month: {ref['year_month']}\n"
            f"excerpt: {ref['excerpt']}\n"
        )
        if used + len(block) > max_chars:
            break
        used += len(block)
        ref_texts.append(block)

    prompt = (
        "あなたは社内文書検索アシスタントです。\n"
        "以下の文書抜粋のみを根拠として回答してください。\n"
        "推測や補完は禁止です。\n"
        "回答本文には、該当の月報を書いた人の名前と、何年何月のファイルかを必ず含めてください。\n"
        "情報が根拠文書にない場合は『不明』と明記してください。\n\n"
        f"[検索クエリ]\n{query}\n\n"
        f"[参考文書]\n{''.join(ref_texts)}"
    )

    answer = client.generate_answer(prompt=prompt, max_tokens=max_tokens)

    print("\n=== 回答 ===")
    print(answer.strip())
    print("\n=== 参照ファイルパス ===")
    used_paths: List[str] = []
    for ref in references[: len(ref_texts)]:
        path = ref["file_path"]
        if path and path not in used_paths:
            used_paths.append(path)

    for path in used_paths:
        print(path)




def infer_author_and_year_month(file_path: str, title: str, excerpt: str) -> Tuple[str, str]:
    base = f"{file_path} {title} {excerpt}"

    author = "不明"
    year_month = "不明"

    author_patterns = [
        r"(?:作成者|作成|氏名|担当)[:：]\s*([ぁ-んァ-ン一-龥ー々]{2,40})",
        r"([ぁ-んァ-ン一-龥ー々]{2,40})\s*(?:さん|氏)\s*(?:の|作成)",
        r"/月報/([ぁ-んァ-ン一-龥ー々]{2,40})/",
    ]
    for pat in author_patterns:
        m = re.search(pat, base)
        if m:
            author = m.group(1)
            break

    ym_patterns = [
        r"(20\d{2})[-_/年](0?[1-9]|1[0-2])(?:月)?",
        r"(20\d{2})(0[1-9]|1[0-2])",
    ]
    for pat in ym_patterns:
        m = re.search(pat, base)
        if m:
            y, mo = m.group(1), m.group(2).zfill(2)
            year_month = f"{y}年{mo}月"
            break

    return author, year_month

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="monthly_report_RAG")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build or update index from local HTML folder")
    p_index.add_argument("--root", required=True, help="Root folder containing HTML files")
    p_index.add_argument("--index", required=True, help="Directory where index files are stored")
    p_index.add_argument("--chunk-size", type=int, default=500)
    p_index.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p_index.add_argument("--model", default=DEFAULT_GPT_MODEL)

    p_query = sub.add_parser("query", help="Search index and generate answer with GPT")
    p_query.add_argument("--index", required=True, help="Directory where index files are stored")
    p_query.add_argument("query", help="Natural language query")
    p_query.add_argument("--top-k", type=int, default=30)
    p_query.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p_query.add_argument("--model", default=DEFAULT_GPT_MODEL)
    p_query.add_argument("--max-tokens", type=int, default=1200)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command == "index":
        build_index(
            root=Path(args.root),
            index_dir=Path(args.index),
            chunk_size=args.chunk_size,
            embedding_model=args.embedding_model,
            gpt_model=args.model,
        )
        return 0

    if args.command == "query":
        query_index(
            index_dir=Path(args.index),
            query=args.query,
            top_k=args.top_k,
            limit_docs=10,
            embedding_model=args.embedding_model,
            gpt_model=args.model,
            max_tokens=args.max_tokens,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
