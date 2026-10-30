"""Defense question store: source cache, excerpt checks, drill schedule, overview.

All deterministic work of the defense-prep skill goes through this script, so that
excerpt checks and scheduling never depend on Claude's judgement.
Run: conda run -n GraduationProject python defense/tools/defense_db.py <command> ...
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEF = ROOT / "defense"
DATA = Path(os.environ.get("DEFENSE_DATA", DEF / "data"))  # override only for tests
SOURCES = DEF / "sources"
LIT = ROOT / "literature" / "thesis-referenced"
PREFIX = {"claims": "C", "evidence": "E", "questions": "Q"}
TYPES = ["discussion", "considered", "explain", "claim", "fact"]  # fact: rapid-fire warm-up, outside the mix
# HEURISTIC: Leitner-style intervals in days for a 19-day preparation window
NEXT_DAYS = {"wrong": 0, "shaky": 2, "solid": 5}
CONTEXT = 300  # characters of context printed around a matched excerpt


def path(table):
    return DATA / f"{table}.jsonl"


def load(table):
    p = path(table)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(table, records):
    DATA.mkdir(parents=True, exist_ok=True)
    with path(table).open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rewrite(table, records):
    DATA.mkdir(parents=True, exist_ok=True)
    path(table).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def next_id(table):
    n = max((int(r["id"].split("-")[1]) for r in load(table)), default=0)
    return f"{PREFIX[table]}-{n + 1:03d}"


def norm(text):
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------- source cache and checks ----------

def pdf_pages(pdf):
    """Text per page; PyMuPDF first (handles encrypted PDFs), pypdf as fallback."""
    try:
        import fitz
        with fitz.open(str(pdf)) as doc:
            return [page.get_text() for page in doc]
    except ImportError:
        from pypdf import PdfReader
        return [page.extract_text() or "" for page in PdfReader(str(pdf)).pages]


def cmd_sources(a):
    SOURCES.mkdir(parents=True, exist_ok=True)
    pdfs = [Path(p) for p in a.pdf] if a.pdf else sorted(LIT.glob("*.pdf"))
    failed = []
    for pdf in pdfs:
        out = SOURCES / f"{pdf.stem}.txt"
        if out.exists() and not a.force:
            print(f"skip {pdf.stem} (cached)")
            continue
        try:
            pages = pdf_pages(pdf)
        except Exception as exc:
            failed.append(pdf.stem)
            print(f"FAILED {pdf.stem}: {exc}")
            continue
        out.write_text("\n".join(f"=== p.{i + 1} ===\n{t}" for i, t in enumerate(pages)), encoding="utf-8")
        print(f"cached {pdf.stem}: {len(pages)} pages")
    if failed:
        sys.exit(f"{len(failed)} PDF(s) not cached: {', '.join(failed)}")


def csv_rows(source):
    import csv
    with (ROOT / source[4:]).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def page_text(source, page):
    """PDF source: text of PDF page `page`. tex:<path>: lines page-40 to page+40, LaTeX comments removed.
    csv:<path from repo root>: data row `page` (1-based) as `col=value; ...`."""
    if source.startswith("csv:"):
        rows = csv_rows(source)
        if not 1 <= page <= len(rows):
            sys.exit(f"{source} has no data row {page} (1 to {len(rows)})")
        return "; ".join(f"{k}={v}" for k, v in rows[page - 1].items())
    if source.startswith("tex:"):
        lines = (ROOT / "Thesis-writeup" / "Writing" / source[4:]).read_text(encoding="utf-8").splitlines()
        return "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in lines[max(0, page - 41):page + 40])
    cache = SOURCES / f"{source}.txt"
    if not cache.exists():
        sys.exit(f"no cache for {source}: run `sources --pdf <file>` first")
    chunks = re.split(r"^=== p\.(\d+) ===$", cache.read_text(encoding="utf-8"), flags=re.M)
    pages = {int(chunks[i]): chunks[i + 1] for i in range(1, len(chunks), 2)}
    if page not in pages:
        sys.exit(f"{source} has no PDF page {page} (1 to {max(pages)})")
    return pages[page]


def check(source, page, excerpt):
    flat, want = norm(page_text(source, page)), norm(excerpt)
    i = flat.find(want)
    if i < 0:
        return False, ""
    return True, flat[max(0, i - CONTEXT):i + len(want) + CONTEXT]


def cmd_check(a):
    ok, ctx = check(a.source, a.page, a.excerpt)
    if not ok:
        sys.exit(f"MISMATCH: excerpt not on {a.source} p.{a.page}")
    print(f"MATCH OK {a.source} p.{a.page}\ncontext: ...{ctx}...")


def cmd_recheck(a):
    ev = {e["id"]: e for e in load("evidence")}
    bad = 0
    for eid in a.ids:
        e = ev.get(eid)
        if e is None:
            bad += 1
            print(f"{eid}: unknown")
            continue
        ok, ctx = check(e["source"], e["pdf_page"], e["excerpt"])
        bad += not ok
        print(f"{eid} {'MATCH OK' if ok else 'MISMATCH'} {e['source']} p.{e['pdf_page']}: ...{ctx[:400]}...")
    sys.exit(1 if bad else 0)


def cmd_rows(a):
    rows = csv_rows(a.source)
    filters = [f.split("=", 1) for f in a.where]
    for i, r in enumerate(rows, 1):
        if all(r.get(k) == v for k, v in filters):
            print(f"row {i}: " + "; ".join(f"{k}={v}" for k, v in r.items()))


def cmd_annots(a):
    from pypdf import PdfReader
    for i, page in enumerate(PdfReader(a.pdf).pages):
        for ref in page.get("/Annots") or []:
            an = ref.get_object()
            text = an.get("/Contents")
            if text:
                print(f"p.{i + 1} [{an.get('/T', '')}] {str(text).strip()}")


# ---------- records ----------

def cmd_add_evidence(a):
    ok, _ = check(a.source, a.page, a.excerpt)
    if not ok:
        sys.exit(f"MISMATCH: excerpt not on {a.source} p.{a.page}; evidence not saved")
    rec = {"id": next_id("evidence"), "source": a.source, "pdf_page": a.page, "printed_page": a.printed,
           "excerpt": a.excerpt, "claim": a.claim, "kind": a.kind, "checked": str(dt.date.today())}
    append("evidence", [rec])
    print(f"saved {rec['id']}")


def key(q):
    return "|".join([q.get("topic", ""), q.get("type", ""), q.get("voice", "")])


def cmd_add(a):
    recs = [json.loads(line) for line in Path(a.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.table == "questions":
        ev = {e["id"] for e in load("evidence")}
        keys = {key(q) for q in load("questions")}
        for q in recs:
            if q.get("type") not in TYPES:
                sys.exit(f"type must be one of {TYPES}: {q.get('question', '')[:60]}")
            missing = [e for e in q.get("evidence", []) if e not in ev]
            if missing:
                sys.exit(f"unknown evidence {missing}: {q.get('question', '')[:60]}")
            if not q.get("evidence") and q.get("grounding") != "outside":
                sys.exit(f"no evidence: add checked evidence or set grounding 'outside': {q.get('question', '')[:60]}")
            if key(q) in keys:
                sys.exit(f"duplicate topic|type|voice {key(q)}: choose another type or voice")
            keys.add(key(q))
            q.setdefault("status", "active")
    saved = []
    for r in recs:
        r.setdefault("id", next_id(a.table))
        append(a.table, [r])
        saved.append(r["id"])
    print(f"saved {len(saved)} to {a.table}: {', '.join(saved)}")


def cmd_drop(a):
    qs = load("questions")
    for q in qs:
        if q["id"] in a.ids:
            q["status"] = "dropped"
            q["drop_reason"] = a.reason
    rewrite("questions", qs)
    print(f"dropped {', '.join(a.ids)}")


# ---------- drill ----------

def cmd_log_drill(a):
    today = dt.date.today()
    prev = [d for d in load("drills") if d["id"] == a.q]
    retire = a.result == "solid" and bool(prev) and prev[-1]["result"] == "solid"
    rec = {"id": a.q, "date": str(today), "mode": a.mode, "result": a.result,
           "gap": a.gap, "next_due": None if retire else str(today + dt.timedelta(days=NEXT_DAYS[a.result]))}
    if a.confidence:
        # optional, never asked of the user (D-290); scheduling uses result only
        rec["confidence"] = a.confidence
    append("drills", [rec])
    print(f"logged {a.q}: {a.result}, " + ("retired from drill" if retire else f"due {rec['next_due']}"))


def due_list(today):
    last = {d["id"]: d for d in load("drills")}
    out = []
    for q in load("questions"):
        if q.get("status") != "active":
            continue
        d = last.get(q["id"])
        if d is None:
            out.append((q, "new"))
        elif d["next_due"] and d["next_due"] <= str(today):
            out.append((q, d["result"]))
    order = {"wrong": 0, "shaky": 1, "solid": 2, "new": 3}
    return sorted(out, key=lambda t: order[t[1]])


def cmd_due(a):
    items = [(q, why) for q, why in due_list(dt.date.today()) if not a.type or q.get("type") == a.type]
    counts = {t: sum(1 for q in load("questions") if q.get("status") == "active" and q.get("type") == t)
              for t in TYPES}
    print("active per type: " + ", ".join(f"{t} {n}" for t, n in counts.items()))
    for q, why in items:
        print(f"{q['id']} [{why}] ({q.get('type', '')}, {q.get('voice', '')}) {q['question'][:90]}")


# ---------- checks and overview ----------

def cmd_validate(a):
    errors = 0
    for e in load("evidence"):
        if not check(e["source"], e["pdf_page"], e["excerpt"])[0]:
            errors += 1
            print(f"MISMATCH {e['id']} {e['source']} p.{e['pdf_page']}")
    ev = {e["id"] for e in load("evidence")}
    for q in load("questions"):
        for e in q.get("evidence", []):
            if e not in ev:
                errors += 1
                print(f"{q['id']}: unknown evidence {e}")
        ans = DEF / "answers" / f"{q['id']}.md"
        if ans.exists():
            for e in re.findall(r"\bE-\d{3}\b", ans.read_text(encoding="utf-8")):
                if e not in ev:
                    errors += 1
                    print(f"{ans.name}: unknown evidence {e}")
    print("VALID" if errors == 0 else f"{errors} error(s)")
    sys.exit(1 if errors else 0)


def cmd_overview(a):
    today = dt.date.today()
    last = {d["id"]: d for d in load("drills")}
    qs = [q for q in load("questions") if q.get("status") == "active"]
    lines = ["# Defense question overview", "", f"Generated {today} by `defense_db.py overview`; do not edit.", "",
             f"Active questions {len(qs)} | evidence {len(load('evidence'))} | due today {len(due_list(today))}",
             "", "| id | type | voice | last result | next due | answer | question |", "|-|-|-|-|-|-|-|"]
    for q in qs:
        d = last.get(q["id"], {})
        has_answer = "yes" if (DEF / "answers" / f"{q['id']}.md").exists() else ""
        text = q["question"].replace("|", "/")[:80]
        lines.append(f"| {q['id']} | {q.get('type', '')} | {q.get('voice', '')} | {d.get('result', '')} | "
                     f"{d.get('next_due', '') or ''} | {has_answer} | {text} |")
    out = DATA.parent / "overview.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(qs)} active questions)")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sources", help="cache PDFs as page-tagged text in defense/sources/")
    s.add_argument("--pdf", nargs="*", help="PDF paths (default: literature/thesis-referenced/*.pdf)")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("check", help="check a verbatim excerpt on a source page")
    s.add_argument("source", help="cite key (cache name) or tex:sections/<file>.tex")
    s.add_argument("page", type=int, help="PDF page, or line number for tex:")
    s.add_argument("excerpt")
    s = sub.add_parser("recheck", help="re-check saved evidence records before use")
    s.add_argument("ids", nargs="+")
    s = sub.add_parser("rows", help="print numbered data rows of a results CSV, optionally filtered")
    s.add_argument("source", help="csv:<path from repo root>")
    s.add_argument("--where", nargs="*", default=[], help="col=value filters")
    s = sub.add_parser("annots", help="print the comments in an annotated PDF")
    s.add_argument("pdf")
    s = sub.add_parser("add-evidence", help="check an excerpt and save it as evidence")
    s.add_argument("--source", required=True)
    s.add_argument("--page", type=int, required=True)
    s.add_argument("--printed", default="", help="printed page number, if any")
    s.add_argument("--excerpt", required=True)
    s.add_argument("--claim", required=True, help="the claim this excerpt supports")
    s.add_argument("--kind", choices=["direct", "derived"], required=True)
    s = sub.add_parser("add", help="append JSONL records from a file")
    s.add_argument("table", choices=["claims", "questions"])
    s.add_argument("file")
    s = sub.add_parser("drop", help="drop questions the user rejects")
    s.add_argument("ids", nargs="+")
    s.add_argument("--reason", default="")
    s = sub.add_parser("log-drill", help="log a result and schedule the next ask")
    s.add_argument("--q", required=True)
    s.add_argument("--mode", choices=["practice", "chain", "mock", "study"], default="practice")
    s.add_argument("--confidence", choices=["sure", "unsure"], help="optional; not used for scheduling")
    s.add_argument("--result", choices=["solid", "shaky", "wrong"], required=True)
    s.add_argument("--gap", default="")
    s = sub.add_parser("due", help="questions due, weakest first, with counts per type")
    s.add_argument("--type", choices=TYPES, help="only this type (fact for the warm-up)")
    sub.add_parser("validate", help="re-check every excerpt and reference")
    sub.add_parser("overview", help="write defense/overview.md")
    a = p.parse_args()
    {"sources": cmd_sources, "check": cmd_check, "recheck": cmd_recheck, "rows": cmd_rows, "annots": cmd_annots,
     "add-evidence": cmd_add_evidence, "add": cmd_add, "drop": cmd_drop, "log-drill": cmd_log_drill,
     "due": cmd_due, "validate": cmd_validate, "overview": cmd_overview}[a.cmd](a)


if __name__ == "__main__":
    main()
