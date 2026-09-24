"""G0(c): every vendored file differs from its source only in lines marked VENDOR-PATCH (JH-001).

Reads the provenance table in vendor/VENDORED.md. An added line must carry the marker; a removed
source line is allowed only inside a hunk that also adds marked lines (a patched replacement).
Also reports whether the SOURCE changed since the copy (sha256 against the table).
"""
import difflib
import hashlib
import os
import re
import sys

JH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(JH, '..', '..', '..'))
MARKS = ('VENDOR-PATCH (JH-001)', 'VENDOR-PATCH (JH-004)', 'VENDOR-PATCH (JH-007)')

rows = []
for ln in open(os.path.join(JH, 'vendor', 'VENDORED.md'), encoding='utf-8'):
    m = re.match(r'\| `([^`]+)` \| ([0-9a-f]{12}) \| (\S+) \| `([^`]+)` \|', ln)
    if m:
        rows.append(m.groups())
n_unmarked, n_marked, n_src_changed = 0, 0, 0
for src, sha, status, copy in rows:
    s_path, c_path = os.path.join(REPO, src), os.path.join(JH, copy)
    if s_path.endswith(('.npz', '.mat')):                 # binary: byte-identical or unmarked
        now = hashlib.sha256(open(s_path, 'rb').read()).hexdigest()[:12]
        if open(s_path, 'rb').read() != open(c_path, 'rb').read():
            n_unmarked += 1
            print(f'  BINARY DIFFERS: {copy}')
        if now != sha:
            n_src_changed += 1
            print(f'  SOURCE CHANGED since copy: {src}')
        continue
    s = open(s_path, encoding='utf-8').read().splitlines()
    c = open(c_path, encoding='utf-8').read().splitlines()
    now = hashlib.sha256(open(s_path, 'rb').read()).hexdigest()[:12]
    if now != sha:
        n_src_changed += 1
        print(f'  SOURCE CHANGED since copy: {src} ({sha} -> {now})')
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s, c, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        added = c[j1:j2]
        marked = [a for a in added if any(m in a for m in MARKS)]
        n_marked += len(marked)
        bad = [a for a in added if not any(m in a for m in MARKS) and a.strip()]
        if tag == 'delete' or (i2 > i1 and not marked):
            bad += ['(removed) ' + x for x in s[i1:i2]]
        if bad:
            n_unmarked += len(bad)
            print(f'  UNMARKED in {copy}:')
            for b in bad:
                print('     ', b)
print(f'check_vendor: {len(rows)} files, {n_marked} marked patch lines, {n_unmarked} unmarked '
      f'differences, {n_src_changed} sources changed since the copy')
sys.exit(1 if n_unmarked else 0)
