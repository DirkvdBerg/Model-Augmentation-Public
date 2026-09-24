"""Syntax check of every session script (no imports executed, nothing written: compile() only)."""
import glob
import os
import sys

JH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = [f for pat in ('*.py', 'tools/*.py', 'info/*.py', 'staged/*.py', 'runners/*.py')
         for f in glob.glob(os.path.join(JH, pat))]
bad = 0
for f in sorted(files):
    try:
        compile(open(f, encoding='utf-8').read(), f, 'exec')
    except SyntaxError as e:
        bad += 1
        print('SYNTAX ERROR', os.path.relpath(f, JH), e)
print('compile_all: %d files, %d errors' % (len(files), bad))
sys.exit(1 if bad else 0)
