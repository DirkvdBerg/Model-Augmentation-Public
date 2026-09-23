"""Byte-compile every telica-real Python file (syntax check after edits)."""
import os
import py_compile
import sys

TR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = 0
n = 0
for root, dirs, files in os.walk(TR):
    if 'outputs' in root.split(os.sep) or '__pycache__' in root:
        continue
    for f in files:
        if f.endswith('.py'):
            n += 1
            try:
                py_compile.compile(os.path.join(root, f), doraise=True)
            except py_compile.PyCompileError as e:
                bad += 1
                print('FAIL', e)
print(f'compiled {n} files, {bad} failures')
sys.exit(1 if bad else 0)
