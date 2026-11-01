"""Comments-only check of the 24 grey-box files against the final-step5a snapshot (D-284 amendment 11).

Per file: the syntax tree with every docstring removed must equal the snapshot's (ast.dump, exact). Comments do not
enter the tree. Also prints the comment and docstring line counts before and after.
"""
import ast
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'Clean'
OLD = ROOT / 'simulations/clean-golden/src-snapshots/final-step5a/Clean'
MANIFEST = ROOT / 'simulations/clean-golden/src-snapshots/final-step5a/MANIFEST.sha256'


def stripped_dump(src):
    """ast.dump of src with the docstring of every module, class and function removed."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def comment_lines(src):
    """(lines holding a # comment, lines inside a docstring)."""
    com = {t.start[0] for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.COMMENT}
    doc = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                doc.update(range(b[0].lineno, b[0].end_lineno + 1))
    return len(com), len(doc)


files = [ln.split('*', 1)[1].strip() for ln in MANIFEST.read_text().splitlines() if ln.strip()]
ok = 0
for f in files:
    old, new = (OLD / f).read_text(encoding='utf-8'), (NEW / f).read_text(encoding='utf-8')
    same = stripped_dump(old) == stripped_dump(new)
    ok += same
    (c0, d0), (c1, d1) = comment_lines(old), comment_lines(new)
    print(f"{'IDENTICAL' if same else 'DIFFERENT'} {f}  comments {c0} -> {c1}, docstring lines {d0} -> {d1}")
print(f"AST_IDENTICAL {ok}/{len(files)}" if ok == len(files) else f"AST_DIFFERENT {len(files) - ok}/{len(files)}")
