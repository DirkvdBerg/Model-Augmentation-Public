# Citation Verification Tool

`citecheck.py` verifies that quoted evidence really exists in a source document.
It writes all outputs to `Verified` by default.

The trust rule is:

> The LLM may explain claims, but only `citecheck.py` is trusted to accept quotes.

## Commands

Extract all pages from a PDF or text file:

```powershell
python "literature Verification/citecheck.py" extract --source "path/to/paper.pdf"
```

Find candidate snippets from the document text:

```powershell
python "literature Verification/citecheck.py" find --source "path/to/paper.pdf" --query "frequency resolution"
```

Verify a quote:

```powershell
python "literature Verification/citecheck.py" verify --source "path/to/paper.pdf" --quote "Exact quote here"
```

Verify a quote and save it as an evidence record:

```powershell
python "literature Verification/citecheck.py" add --source "path/to/paper.pdf" --quote "Exact quote here" --claim "Short claim in your own words."
```

Re-check all saved evidence records:

```powershell
python "literature Verification/citecheck.py" audit
```

## Output

Generated files are written to `Verified`, including:

- `*_pages.json` and `*_pages/page_001.txt` files from `extract`
- `find_*.json` and `find_*.md` from `find`
- `verify_*.json` and `verify_*.md` from `verify`
- `evidence_records.json` from `add`
- `audit_*.json` and `audit_*.md` from `audit`

Each accepted quote records:

- source path
- source SHA-256 hash
- page number
- character range on that extracted page
- match mode
- exact matched text from the source extraction
- citation ID

## Match Modes

`exact` means the quote was found exactly in the extracted text.

`normalized-whitespace` means the words and punctuation matched after collapsing
PDF whitespace. This helps with line breaks and column formatting, while still
returning the matched text from the extracted source.

Use `--raw-only` with `verify` or `add` if you want to forbid whitespace
normalization.
