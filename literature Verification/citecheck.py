#!/usr/bin/env python
"""Deterministic citation evidence checker.

This tool keeps the LLM out of the trusted quote path:

- text is extracted directly from a PDF or text file;
- a quote is accepted only if the script can locate it in that extracted text;
- accepted evidence is written with source, page, character offsets, and text.

It uses Poppler's ``pdftotext`` for PDFs. On this machine it is already
available; on another machine, install Poppler if PDF extraction fails.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_OUTPUT_DIR = Path("Verified")
DEFAULT_RECORDS_FILE = DEFAULT_OUTPUT_DIR / "evidence_records.json"


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class MatchResult:
    status: str
    source: str
    source_sha256: str
    quote: str
    page: int | None = None
    start: int | None = None
    end: int | None = None
    match_mode: str | None = None
    matched_text: str | None = None
    citation_id: str | None = None
    closest_matches: list[dict] | None = None
    message: str | None = None


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug or "source"


def make_citation_id(source: Path, page: int, start: int, end: int, quote: str) -> str:
    quote_digest = hashlib.sha1(quote.encode("utf-8")).hexdigest()[:8]
    return f"{slugify(source.stem)}_p{page:03d}_c{start:06d}_{end:06d}_{quote_digest}"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_pages(source: Path) -> list[PageText]:
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if source.suffix.lower() == ".pdf":
        return extract_pdf_pages(source)

    text = source.read_text(encoding="utf-8", errors="replace")
    return [PageText(page=1, text=text)]


def extract_pdf_pages(pdf: Path) -> list[PageText]:
    if shutil.which("pdftotext") is None:
        raise RuntimeError(
            "pdftotext was not found. Install Poppler, then rerun this command."
        )

    command = ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")

    raw_pages = completed.stdout.split("\f")
    pages: list[PageText] = []
    for index, page_text in enumerate(raw_pages, start=1):
        if index == len(raw_pages) and not page_text.strip():
            continue
        pages.append(PageText(page=index, text=page_text.rstrip("\n\r")))
    return pages


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace while retaining normalized-char to raw-char mapping."""
    normalized: list[str] = []
    mapping: list[int] = []
    pending_space_index: int | None = None

    for raw_index, char in enumerate(text):
        if char.isspace():
            if normalized and normalized[-1] != " ":
                pending_space_index = raw_index
            continue

        if pending_space_index is not None:
            normalized.append(" ")
            mapping.append(pending_space_index)
            pending_space_index = None

        normalized.append(char)
        mapping.append(raw_index)

    if normalized and normalized[-1] == " ":
        normalized.pop()
        mapping.pop()

    return "".join(normalized), mapping


def find_raw_match(pages: Iterable[PageText], quote: str) -> tuple[PageText, int, int] | None:
    for page in pages:
        start = page.text.find(quote)
        if start >= 0:
            return page, start, start + len(quote)
    return None


def find_normalized_match(
    pages: Iterable[PageText], quote: str
) -> tuple[PageText, int, int, str] | None:
    normalized_quote, _ = normalize_with_map(quote)
    if not normalized_quote:
        return None

    for page in pages:
        normalized_page, mapping = normalize_with_map(page.text)
        normalized_start = normalized_page.find(normalized_quote)
        if normalized_start >= 0:
            normalized_end = normalized_start + len(normalized_quote)
            raw_start = mapping[normalized_start]
            raw_end = mapping[normalized_end - 1] + 1
            return page, raw_start, raw_end, page.text[raw_start:raw_end]
    return None


def closest_matches(pages: list[PageText], quote: str, limit: int = 5) -> list[dict]:
    normalized_quote, _ = normalize_with_map(quote)
    quote_len = max(len(normalized_quote), 40)
    results: list[dict] = []

    for page in pages:
        normalized_page, mapping = normalize_with_map(page.text)
        if not normalized_page:
            continue

        step = max(20, quote_len // 3)
        window_len = min(max(quote_len + 80, 180), max(len(normalized_page), 1))
        starts = range(0, max(len(normalized_page) - window_len + 1, 1), step)

        for normalized_start in starts:
            candidate = normalized_page[normalized_start : normalized_start + window_len]
            ratio = difflib.SequenceMatcher(None, normalized_quote, candidate).ratio()
            raw_start = mapping[normalized_start] if mapping else 0
            normalized_end = min(normalized_start + window_len, len(mapping)) - 1
            raw_end = mapping[normalized_end] + 1 if mapping and normalized_end >= 0 else 0
            results.append(
                {
                    "page": page.page,
                    "similarity": round(ratio, 3),
                    "start": raw_start,
                    "end": raw_end,
                    "text": compact_whitespace(page.text[raw_start:raw_end]),
                }
            )

    results.sort(key=lambda item: item["similarity"], reverse=True)
    return results[:limit]


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def verify_quote(source: Path, quote: str, allow_normalized: bool = True) -> MatchResult:
    pages = extract_pages(source)
    sha = source_hash(source)

    raw_match = find_raw_match(pages, quote)
    if raw_match is not None:
        page, start, end = raw_match
        matched_text = page.text[start:end]
        return MatchResult(
            status="PASS",
            source=str(source),
            source_sha256=sha,
            quote=quote,
            page=page.page,
            start=start,
            end=end,
            match_mode="exact",
            matched_text=matched_text,
            citation_id=make_citation_id(source, page.page, start, end, matched_text),
        )

    if allow_normalized:
        normalized_match = find_normalized_match(pages, quote)
        if normalized_match is not None:
            page, start, end, matched_text = normalized_match
            return MatchResult(
                status="PASS",
                source=str(source),
                source_sha256=sha,
                quote=quote,
                page=page.page,
                start=start,
                end=end,
                match_mode="normalized-whitespace",
                matched_text=matched_text,
                citation_id=make_citation_id(source, page.page, start, end, matched_text),
            )

    return MatchResult(
        status="FAIL",
        source=str(source),
        source_sha256=sha,
        quote=quote,
        closest_matches=closest_matches(pages, quote),
        message="Quote not found in extracted text.",
    )


def read_quote(args: argparse.Namespace) -> str:
    if args.quote:
        return args.quote
    if args.quote_file:
        return Path(args.quote_file).read_text(encoding="utf-8")
    raise SystemExit("Provide --quote or --quote-file.")


def write_json(path: Path, payload: object) -> None:
    ensure_output_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_output_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def render_match_markdown(result: MatchResult, claim: str | None = None) -> str:
    lines = [
        f"# Citation Verification: {result.status}",
        "",
        f"- Source: `{result.source}`",
        f"- Source SHA-256: `{result.source_sha256}`",
    ]
    if claim:
        lines.append(f"- Claim: {claim}")

    if result.status == "PASS":
        lines.extend(
            [
                f"- Citation ID: `{result.citation_id}`",
                f"- Page: {result.page}",
                f"- Character range: {result.start}-{result.end}",
                f"- Match mode: {result.match_mode}",
                "",
                "## Verified Quote",
                "",
                blockquote(result.matched_text or ""),
            ]
        )
    else:
        lines.extend(["", f"## Failure", "", result.message or "Quote not found."])
        if result.closest_matches:
            lines.extend(["", "## Closest Matches", ""])
            for item in result.closest_matches:
                lines.extend(
                    [
                        f"### Page {item['page']} similarity {item['similarity']}",
                        "",
                        blockquote(item["text"]),
                        "",
                    ]
                )

    return "\n".join(lines).rstrip() + "\n"


def blockquote(text: str) -> str:
    compact = text.strip() or "(empty)"
    return "\n".join(f"> {line}" for line in compact.splitlines())


def command_extract(args: argparse.Namespace) -> int:
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    pages = extract_pages(source)
    sha = source_hash(source)
    stem = slugify(source.stem)
    page_records = [asdict(page) for page in pages]
    payload = {
        "source": str(source),
        "source_sha256": sha,
        "page_count": len(pages),
        "pages": page_records,
    }

    json_path = output_dir / f"{stem}_pages.json"
    write_json(json_path, payload)

    page_text_dir = output_dir / f"{stem}_pages"
    ensure_output_dir(page_text_dir)
    for page in pages:
        write_text(page_text_dir / f"page_{page.page:03d}.txt", page.text)

    print(f"PASS extracted {len(pages)} page(s)")
    print(f"JSON: {json_path}")
    print(f"Text pages: {page_text_dir}")
    return 0


def command_find(args: argparse.Namespace) -> int:
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    pages = extract_pages(source)
    query, _ = normalize_with_map(args.query)
    context = args.context
    snippets: list[dict] = []

    for page in pages:
        normalized_page, mapping = normalize_with_map(page.text)
        search_start = 0
        while query and True:
            hit = normalized_page.lower().find(query.lower(), search_start)
            if hit < 0:
                break
            raw_start = mapping[max(0, hit - context)]
            raw_end = mapping[min(len(mapping) - 1, hit + len(query) + context)] + 1
            snippet_text = page.text[raw_start:raw_end]
            snippet_id = make_citation_id(source, page.page, raw_start, raw_end, snippet_text)
            snippets.append(
                {
                    "id": snippet_id,
                    "source": str(source),
                    "source_sha256": source_hash(source),
                    "page": page.page,
                    "start": raw_start,
                    "end": raw_end,
                    "query": args.query,
                    "text": snippet_text,
                }
            )
            search_start = hit + len(query)
            if len(snippets) >= args.limit:
                break
        if len(snippets) >= args.limit:
            break

    stamp = now_stamp()
    json_path = output_dir / f"find_{stamp}.json"
    md_path = output_dir / f"find_{stamp}.md"
    write_json(json_path, {"query": args.query, "snippets": snippets})
    write_text(md_path, render_snippets_markdown(args.query, snippets))

    print(f"PASS found {len(snippets)} snippet(s)")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0 if snippets else 1


def render_snippets_markdown(query: str, snippets: list[dict]) -> str:
    lines = [f"# Evidence Search: {query}", ""]
    if not snippets:
        lines.append("No snippets found.")
        return "\n".join(lines) + "\n"

    for snippet in snippets:
        lines.extend(
            [
                f"## {snippet['id']}",
                "",
                f"- Source: `{snippet['source']}`",
                f"- Page: {snippet['page']}",
                f"- Character range: {snippet['start']}-{snippet['end']}",
                "",
                blockquote(snippet["text"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def command_verify(args: argparse.Namespace) -> int:
    source = Path(args.source)
    quote = read_quote(args)
    result = verify_quote(source, quote, allow_normalized=not args.raw_only)
    stamp = now_stamp()
    output_dir = Path(args.output_dir)
    json_path = output_dir / f"verify_{stamp}.json"
    md_path = output_dir / f"verify_{stamp}.md"
    write_json(json_path, asdict(result))
    write_text(md_path, render_match_markdown(result, claim=args.claim))

    print(result.status)
    if result.status == "PASS":
        print(f"Citation ID: {result.citation_id}")
        print(f"Page: {result.page}")
        print(f"Character range: {result.start}-{result.end}")
        print(f"Match mode: {result.match_mode}")
    else:
        print(result.message)
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0 if result.status == "PASS" else 1


def command_add(args: argparse.Namespace) -> int:
    source = Path(args.source)
    quote = read_quote(args)
    result = verify_quote(source, quote, allow_normalized=not args.raw_only)
    output_dir = Path(args.output_dir)
    stamp = now_stamp()

    json_path = output_dir / f"add_{stamp}.json"
    md_path = output_dir / f"add_{stamp}.md"
    write_json(json_path, asdict(result))
    write_text(md_path, render_match_markdown(result, claim=args.claim))

    if result.status != "PASS":
        print("FAIL")
        print(result.message)
        print(f"JSON: {json_path}")
        print(f"Markdown: {md_path}")
        return 1

    records_path = Path(args.records)
    records = load_records(records_path)
    record = asdict(result)
    record["claim"] = args.claim
    record["added_at"] = datetime.now().isoformat(timespec="seconds")
    records = [item for item in records if item.get("citation_id") != result.citation_id]
    records.append(record)
    write_json(records_path, records)

    print("PASS")
    print(f"Added: {result.citation_id}")
    print(f"Records: {records_path}")
    print(f"Markdown: {md_path}")
    return 0


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of records in {path}")
    return payload


def command_audit(args: argparse.Namespace) -> int:
    records_path = Path(args.records)
    records = load_records(records_path)
    audited: list[dict] = []
    failures = 0

    for record in records:
        source = Path(record["source"])
        result = verify_quote(source, record["matched_text"], allow_normalized=False)
        audit_item = asdict(result)
        audit_item["original_citation_id"] = record.get("citation_id")
        audit_item["claim"] = record.get("claim")
        audit_item["expected_source_sha256"] = record.get("source_sha256")
        audit_item["source_hash_matches"] = (
            result.source_sha256 == record.get("source_sha256")
        )
        if result.status != "PASS" or not audit_item["source_hash_matches"]:
            failures += 1
        audited.append(audit_item)

    stamp = now_stamp()
    output_dir = Path(args.output_dir)
    json_path = output_dir / f"audit_{stamp}.json"
    md_path = output_dir / f"audit_{stamp}.md"
    write_json(
        json_path,
        {"records": str(records_path), "total": len(records), "failures": failures, "items": audited},
    )
    write_text(md_path, render_audit_markdown(records_path, audited, failures))

    print("PASS" if failures == 0 else "FAIL")
    print(f"Audited records: {len(records)}")
    print(f"Failures: {failures}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0 if failures == 0 else 1


def render_audit_markdown(records_path: Path, audited: list[dict], failures: int) -> str:
    lines = [
        "# Citation Audit",
        "",
        f"- Records: `{records_path}`",
        f"- Total: {len(audited)}",
        f"- Failures: {failures}",
        "",
    ]
    for item in audited:
        lines.extend(
            [
                f"## {item.get('original_citation_id')}",
                "",
                f"- Status: {item['status']}",
                f"- Source hash matches: {item['source_hash_matches']}",
                f"- Source: `{item['source']}`",
                f"- Page: {item.get('page')}",
                f"- Character range: {item.get('start')}-{item.get('end')}",
            ]
        )
        if item.get("claim"):
            lines.append(f"- Claim: {item['claim']}")
        lines.extend(["", blockquote(item.get("matched_text") or ""), ""])
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract, find, verify, add, and audit source-backed citation evidence."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for generated verification output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract source pages to Verified.")
    extract.add_argument("--source", required=True, help="PDF or UTF-8 text source.")
    extract.set_defaults(func=command_extract)

    find = subparsers.add_parser("find", help="Find source snippets around a query.")
    find.add_argument("--source", required=True, help="PDF or UTF-8 text source.")
    find.add_argument("--query", required=True, help="Search query.")
    find.add_argument("--context", type=int, default=350, help="Context characters.")
    find.add_argument("--limit", type=int, default=20, help="Maximum snippets.")
    find.set_defaults(func=command_find)

    verify = subparsers.add_parser("verify", help="Verify that a quote exists in a source.")
    add_quote_args(verify)
    verify.set_defaults(func=command_verify)

    add = subparsers.add_parser("add", help="Verify and append a record to evidence_records.json.")
    add_quote_args(add)
    add.add_argument(
        "--records",
        default=str(DEFAULT_RECORDS_FILE),
        help=f"Evidence record file. Default: {DEFAULT_RECORDS_FILE}",
    )
    add.set_defaults(func=command_add)

    audit = subparsers.add_parser("audit", help="Re-verify all saved evidence records.")
    audit.add_argument(
        "--records",
        default=str(DEFAULT_RECORDS_FILE),
        help=f"Evidence record file. Default: {DEFAULT_RECORDS_FILE}",
    )
    audit.set_defaults(func=command_audit)

    return parser


def add_quote_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, help="PDF or UTF-8 text source.")
    parser.add_argument("--quote", help="Exact quote to verify.")
    parser.add_argument("--quote-file", help="Text file containing the quote to verify.")
    parser.add_argument("--claim", help="Optional claim this quote supports.")
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Require byte-for-byte text match after extraction; do not normalize whitespace.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
