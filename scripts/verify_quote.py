"""
verify_quote.py -- Code quote verifier for Claude Code sessions.

Usage:
    python scripts/verify_quote.py <file> <start_line> <end_line> <quote_file>

Arguments:
    file        Path to the source file to verify against
    start_line  First line (1-based, inclusive)
    end_line    Last line (1-based, inclusive)
    quote_file  Path to file containing the quote to verify

Output:
    Prints the actual file lines with metadata, then MATCH OK or MISMATCH with diff.
    Exit code 0 = match, 1 = mismatch or error.

Comparison uses normalized lines (trailing whitespace stripped) so minor
formatting differences do not cause false mismatches.
The SHA256 hash is of the raw section bytes -- re-run later to confirm the
file has not changed since the quote was verified.
"""

import sys
import os
import hashlib
import difflib

BOX_W = 66

def box_top(): print("+" + "-" * BOX_W + "+")
def box_mid(): print("+" + "-" * BOX_W + "+")
def box_bot(): print("+" + "-" * BOX_W + "+")
def box_row(s=""):
    s = str(s)
    if len(s) > BOX_W - 2:
        s = s[:BOX_W - 5] + "..."
    print("|  " + s.ljust(BOX_W - 2) + "|")

def normalize(lines):
    return [l.rstrip() for l in lines]

def main():
    if len(sys.argv) != 5:
        print("Usage: python verify_quote.py <file> <start_line> <end_line> <quote_file>")
        print("  quote_file : file containing the text you intend to quote")
        sys.exit(1)

    file_path  = sys.argv[1]
    start_line = int(sys.argv[2])
    end_line   = int(sys.argv[3])
    quote_file = sys.argv[4]

    # -- read source file -------------------------------------------------
    if not os.path.exists(file_path):
        print(f"ERROR: file not found: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    total_lines = len(all_lines)

    if start_line < 1 or end_line > total_lines or start_line > end_line:
        print(f"ERROR: line range {start_line}-{end_line} invalid (file has {total_lines} lines)")
        sys.exit(1)

    section_lines = all_lines[start_line - 1 : end_line]
    section_raw   = "".join(section_lines)
    section_hash  = hashlib.sha256(section_raw.encode("utf-8")).hexdigest()[:16]

    from datetime import datetime
    mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M")

    # -- read quote file --------------------------------------------------
    if not os.path.exists(quote_file):
        print(f"ERROR: quote file not found: {quote_file}")
        sys.exit(1)

    with open(quote_file, "r", encoding="utf-8") as f:
        quote_lines = f.readlines()

    # -- print header -----------------------------------------------------
    box_top()
    box_row("SHELL VERIFICATION")
    box_mid()
    box_row(f"File    : {file_path}")
    box_row(f"Lines   : {start_line}-{end_line}  ({total_lines} total, modified {mtime})")
    box_row(f"Hash    : {section_hash}  (SHA256 of raw section bytes, first 16 chars)")
    box_mid()

    for i, line in enumerate(section_lines, start=start_line):
        box_row(f"{i:>4}| {line.rstrip()}")

    # -- compare ----------------------------------------------------------
    box_mid()
    actual_norm = normalize(section_lines)
    quote_norm  = normalize(quote_lines)

    if actual_norm == quote_norm:
        box_row("MATCH OK  -- quote matches file exactly (normalized)")
        box_bot()
        sys.exit(0)
    else:
        box_row("MISMATCH  -- diff below  (--- quote  +++ actual file)")
        box_mid()
        diff = difflib.unified_diff(
            quote_norm, actual_norm,
            fromfile="my_quote", tofile="actual_file",
            lineterm=""
        )
        for dl in list(diff)[2:]:
            box_row(dl)
        box_bot()
        sys.exit(1)

if __name__ == "__main__":
    main()
