"""Byte-compile every Python file of this folder (vendored copies excluded); print failures."""
import os
import py_compile
import sys

CN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad, n = [], 0
for root, dirs, files in os.walk(CN):
    dirs[:] = [d for d in dirs if d not in ('vendor', 'outputs', '__pycache__')]
    for f in files:
        if f.endswith('.py'):
            n += 1
            try:
                py_compile.compile(os.path.join(root, f), doraise=True)
            except py_compile.PyCompileError as exc:
                bad.append(str(exc))
print(f'[compile] {n} files, {len(bad)} failures')
for b in bad:
    print(b)
sys.exit(1 if bad else 0)
