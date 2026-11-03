"""Compare a Clean file of model_augmentation with Jan's original, definition by definition.

Part 1 lists, per function and class, whether the code, the docstring and the comments are the same as Jan's
(code compared without comments, with the D-282 renames mapped). Part 2 (blocks.py only) builds Jan's and the
Clean block with the same weights and inputs and checks that the outputs are identical.
"""
import argparse
import ast
import difflib
import importlib
import io
import re
import sys
import tokenize
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
JAN = ROOT / 'Model-Augmentation-Public-main'
CLEAN = ROOT / 'Clean'

# file inside model_augmentation/ to compare, and whether to print every difference in full
FILE = 'fit_systems/blocks.py'
VERBOSE = False

# Jan's names that Clean renamed (D-282)
RENAMES = {
    'fixed_System_data': 'WindowedSystemData', 'fixed_System_data_norm': 'SystemDataNormalizer',
    'mlp_stack': 'build_mlp', 'Dtype_DataLoader': 'DtypeDataLoader', '_replace': '_replace_with_retry',
    'FixedBatch': 'FixedWindowBatch',
}


def rename(text):
    """Jan's names replaced by the Clean names."""
    for jan_name, clean_name in RENAMES.items():
        text = re.sub(rf'\b{jan_name}\b', clean_name, text)
    return text


def without_docstring(node):
    """Copy of the body of node without its docstring."""
    body = node.body
    if ast.get_docstring(node, clean=False) is not None:
        body = body[1:]
    return body or [ast.Pass()]


def definitions(path):
    """Per definition: (first line, last line, code without comments and docstrings, docstring, comments)."""
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    spans, out = [], {}
    module_code = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            header_code = []
            for member in without_docstring(node):
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    member_copy = ast.parse(ast.unparse(member)).body[0]
                    member_copy.body = without_docstring(member_copy)
                    name = f'{node.name}.{member.name}'
                    start = min([member.lineno] + [d.lineno for d in member.decorator_list])
                    spans.append((name, start, member.end_lineno))
                    out[name] = [start, ast.unparse(member_copy), ast.get_docstring(member) or '', []]
                else:
                    header_code.append(ast.unparse(member))
            bases = ', '.join(ast.unparse(b) for b in node.bases)
            decorators = ''.join('@' + ast.unparse(d) + '\n' for d in node.decorator_list)
            name = f'{node.name}<header>'
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            spans.append((name, start, node.end_lineno))
            out[name] = [start, f'{decorators}class {node.name}({bases})\n' + '\n'.join(header_code),
                         ast.get_docstring(node) or '', []]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node_copy = ast.parse(ast.unparse(node)).body[0]
            node_copy.body = without_docstring(node_copy)
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            spans.append((node.name, start, node.end_lineno))
            out[node.name] = [start, ast.unparse(node_copy), ast.get_docstring(node) or '', []]
        elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                  and isinstance(node.value.value, str)):
            module_code.append(ast.unparse(node))
    out['<module>'] = [1, '\n'.join(module_code), ast.get_docstring(tree) or '', []]
    # each comment belongs to the innermost definition around its line, else to the module
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            row = tok.start[0]
            around = [s for s in spans if s[1] <= row <= s[2]]
            owner = max(around, key=lambda s: s[1])[0] if around else '<module>'
            out[owner][3].append(tok.string.lstrip('#').strip())
    return out


def status(jan, clean):
    """SAME, ADDED, REMOVED or CHANGED for one part of a definition; '-' when both are empty."""
    if jan == clean:
        return 'SAME' if jan else '-'
    if not jan:
        return 'ADDED'
    if not clean:
        return 'REMOVED'
    return 'CHANGED'


def compare_text(file, verbose):
    """Part 1: one line per definition with the status of code, docstring and comments."""
    jan = {rename(k): [line, rename(code), doc, comments]
           for k, (line, code, doc, comments) in definitions(JAN / 'model_augmentation' / file).items()}
    clean = definitions(CLEAN / 'model_augmentation' / file)
    # in both files, in Jan's order; then Jan only; then Clean only
    shared = sorted(set(jan) & set(clean), key=lambda k: jan[k][0])
    removed = sorted(set(jan) - set(clean), key=lambda k: jan[k][0])
    added = sorted(set(clean) - set(jan), key=lambda k: clean[k][0])
    print(f'{"in both (Jan order)":60s} {"code":8s} {"docstring":10s} comments')
    for name in shared:
        _, jan_code, jan_doc, jan_comments = jan[name]
        _, clean_code, clean_doc, clean_comments = clean[name]
        print(f'{name:60s} {status(jan_code, clean_code):8s} {status(jan_doc, clean_doc):10s} '
              f'{status(jan_comments, clean_comments)}   (Jan line {jan[name][0]}, Clean line {clean[name][0]})')
        if verbose:
            if jan_code != clean_code:
                for line in difflib.unified_diff(jan_code.splitlines(), clean_code.splitlines(), lineterm='', n=1):
                    if not line.startswith(('---', '+++')):
                        print('      code ' + line)
            if jan_doc != clean_doc:
                print('      docstring Jan:   ' + (jan_doc.replace('\n', '\n' + ' ' * 23) or '(none)'))
                print('      docstring Clean: ' + (clean_doc.replace('\n', '\n' + ' ' * 23) or '(none)'))
            if jan_comments != clean_comments:
                for c in jan_comments:
                    print('      comment Jan:   # ' + c)
                for c in clean_comments:
                    print('      comment Clean: # ' + c)
    print('\nonly in Jan (removed):')
    for name in removed:
        print(f'  {name:58s} (Jan line {jan[name][0]})')
    print('\nonly in Clean (added):')
    for name in added:
        print(f'  {name:58s} (Clean line {clean[name][0]})')


def load_both(module):
    """Jan's and the Clean version of model_augmentation.<module>, loaded one after the other."""
    loaded = []
    for root in (JAN, CLEAN):
        sys.path.insert(0, str(root))
        loaded.append(importlib.import_module(f'model_augmentation.{module}'))
        sys.path.remove(str(root))
        for key in [k for k in sys.modules if k == 'model_augmentation' or k.startswith('model_augmentation.')]:
            del sys.modules[key]
    return loaded


class SmallNet(nn.Module):
    """Two-layer tanh network with the arguments both Static_ANN_Block versions pass."""

    def __init__(self, n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, n_nodes_per_layer), activation(),
                                    nn.Linear(n_nodes_per_layer, n_out))

    def forward(self, z):
        return self.layers(z)


def check_blocks():
    """Part 2 for blocks.py: same inputs and weights in Jan's and the Clean block, outputs compared exactly."""
    jan, clean = load_both('fit_systems.blocks')
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    results = []

    # network block: same weights, random input
    nz, nw = 9, 6
    jan_ann = jan.Static_ANN_Block(net=SmallNet, n_nodes_per_layer=16, n_hidden_layers=1, activation=nn.Tanh,
                                   nz=nz, nw=nw)
    clean_ann = clean.Static_ANN_Block(nz, nw, SmallNet, 16, 1, nn.Tanh, {})
    clean_ann.load_state_dict(jan_ann.state_dict())
    z = torch.randn(32, nz, 1)
    results.append(('Static_ANN_Block.forward', torch.equal(jan_ann(z), clean_ann(z))))
    clean_ann.out_gate.zero_()
    results.append(('Static_ANN_Block.forward, gate 0 gives 0 (Clean only)', bool((clean_ann(z) == 0).all())))

    # output map: same C and D, random input
    C, D = torch.randn(3, 6), torch.zeros(3, 3)
    jan_out, clean_out = jan.Linear_Output_Block(C=C, D=D), clean.Linear_Output_Block(C=C, D=D)
    z = torch.randn(32, 9, 1)
    results.append(('Linear_Output_Block.forward', torch.equal(jan_out(z), clean_out(z))))
    results.append(('Linear_Output_Block sizes ny, nx, nu, nz, nw',
                    all(getattr(jan_out, a) == getattr(clean_out, a) for a in ('ny', 'nx', 'nu', 'nz', 'nw'))))

    # function block: the same map in both; Clean also receives the matrices and passes them on
    class JanSquare(jan.Discrete_Nonlinear_Function_Block):
        def __init__(self):
            super().__init__(nz=9, nw=9)
            self.nx, self.nu = 6, 3

        def nonlinear_function(self, z):
            return z ** 2

    class CleanSquare(clean.Discrete_Nonlinear_Function_Block):
        def __init__(self):
            super().__init__(nz=9, nw=9)

        def nonlinear_function(self, z, mats):
            return z ** 2 * mats

    z = torch.randn(32, 9, 1)
    results.append(('Discrete_Nonlinear_Function_Block.forward (matrices = 1)',
                    torch.equal(JanSquare()(z), CleanSquare()(z, 1.0))))

    print('\noutputs, Jan vs Clean with the same weights and inputs:')
    for name, ok in results:
        print(f'  {"PASS" if ok else "FAIL"}  {name}')
    return all(ok for _, ok in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default=FILE)
    parser.add_argument('-v', '--verbose', action='store_true', default=VERBOSE)
    args = parser.parse_args()
    compare_text(args.file, args.verbose)
    if args.file == 'fit_systems/blocks.py':
        sys.exit(0 if check_blocks() else 1)
