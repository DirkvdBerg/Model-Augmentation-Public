"""Scoreboard of Clean/ (D-284 step 2, rerun unchanged in step 3; step 4b: the 24 grey-box files of snapshot
`final`, harness not an outside caller): static metrics, branch coverage over the layer-1/2 row configs and the
hfn op counts of perf_diag.py.

    python scoreboard.py static   --label pre-round
    python scoreboard.py coverage --label pre-round [--scratch DIR] [case ...]
    python scoreboard.py report   --label pre-round

Everything is written to simulations/clean-golden/perf/<label>/{static,coverage}/. `coverage` runs every case
in its own process under coverage.py (branch coverage, one data file per case), then combines them; run folders
go to <out>/coverage/runs/ and deepSI dumps to the scratch directory. `report` merges static, coverage and the
op counts into scoreboard.json and findings.tsv.
"""
import argparse
import ast
import collections
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
CLEAN = os.path.join(REPO, 'Clean')
# the grey-box files: the manifest of snapshot `final` (D-284 step 4b; Clean/blackbox is another session's)
MANIFEST = os.path.join(REPO, 'simulations', 'clean-golden', 'src-snapshots', 'final', 'MANIFEST.sha256')
# the harness is ours and follows the code, so it is no outside caller (06c-audit-criteria.md, reason (d))
OUTSIDE = [os.path.join(REPO, 'Thesis-writeup', 'Code', 'Evaluation')]
# the 16 cases of run_l12_all.sh
CASES = ['%d-%s' % (r, n) for n in ('lowpass', 'noisefree') for r in (221, 222, 223, 224, 226, 227, 229, 230)]
# the D-281 inventory method (inventory-claude.md section 0): 0.3 s records cross two epoch boundaries; noise-free
# records are at standstill until 0.495 s, so 0.3 s stops in servo_sigma and they need 0.6 s
TRUNCATE_S = {'lowpass': 0.3, 'noisefree': 0.6}
SMOKE = dict(device='cpu', compile_mode=None, n_its=4, its_per_val=4, lbfgs_inner_iter=1, lbfgs_max_iter=2,
             lbfgs_windows=256, ps2_smoke=True)
PERF_ROWS = (226, 221)


def out_dir(label, part):
    d = os.path.join(REPO, 'simulations', 'clean-golden', 'perf', label, part)
    os.makedirs(d, exist_ok=True)
    return d


def py_files():
    with open(MANIFEST, encoding='utf-8') as fh:
        return [os.path.join(CLEAN, *ln.split('*', 1)[1].strip().split('/')) for ln in fh if ln.strip()]


def rel(p):
    return os.path.relpath(p, REPO).replace(os.sep, '/')


def tool(*args):
    """stdout of a tool of the conda environment (this interpreter's -m)."""
    return subprocess.run((sys.executable, '-m') + args, cwd=REPO, capture_output=True, text=True).stdout


# static part

def functions(path):
    """Every def in `path`: qualified name, first and last line, docstring-free body, AST node."""
    tree = ast.parse(open(path, encoding='utf-8').read())
    out = []

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = ch.body[1:] if (ch.body and isinstance(ch.body[0], ast.Expr)
                                       and isinstance(ch.body[0].value, ast.Constant)
                                       and isinstance(ch.body[0].value.value, str)) else ch.body
                out.append(dict(name=ch.name, qual=prefix + ch.name, start=ch.lineno, end=ch.end_lineno,
                                body=body, node=ch))
                walk(ch, prefix + ch.name + '.')
            elif isinstance(ch, ast.ClassDef):
                walk(ch, prefix + ch.name + '.')
            else:
                walk(ch, prefix)
    walk(tree, '')
    return tree, out


def single_call(body):
    """True when the body is one statement whose value is a call (return f(...), f(...), x = f(...))."""
    if len(body) != 1:
        return False
    s = body[0]
    v = getattr(s, 'value', None)
    return isinstance(s, (ast.Return, ast.Expr, ast.Assign)) and isinstance(v, ast.Call)


def passthrough_scan():
    """Pass-through, once-called and never-called functions of Clean/, by name, with outside use."""
    defs, calls, refs = [], collections.Counter(), collections.Counter()
    for p in py_files():
        tree, fns = functions(p)
        for f in fns:
            f['file'] = rel(p)
        defs += fns
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fn = n.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if name:
                    calls[name] += 1
            # a name used without a call (callback, torch.compile argument, decorator, dict entry)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                refs[n.id] += 1
            elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
                refs[n.attr] += 1
    outside_text = {}
    for d in OUTSIDE:
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith('.py') and os.path.abspath(os.path.join(root, f)) != os.path.abspath(__file__):
                    outside_text[rel(os.path.join(root, f))] = open(os.path.join(root, f), encoding='utf-8').read()
    rows = []
    for f in defs:
        n = f['name']
        dunder = n.startswith('__') and n.endswith('__')
        n_calls = calls[n]
        n_refs = refs[n] - n_calls      # the callee of each call is itself a load of the name
        outside = sorted(k for k, t in outside_text.items() if re.search(r'\b%s\b' % re.escape(n), t))
        kinds = []
        if single_call(f['body']):
            kinds.append('pass-through')
        if not dunder and n_calls == 1 and n_refs == 0:
            kinds.append('called once')
        if not dunder and n_calls == 0:
            kinds.append('never called' + (' (referenced %d x)' % n_refs if n_refs else ''))
        if kinds:
            rows.append(dict(file=f['file'], line=f['start'], function=f['qual'], kinds=kinds, calls=n_calls,
                             refs=n_refs, outside=outside))
    return rows


def pylint_duplicates():
    txt = tool('pylint', '--disable=all', '--enable=duplicate-code', '--score=n', *py_files())
    blocks = []
    for m in re.finditer(r'R0801: Similar lines in (\d+) files\n((?:==.*\n)+)', txt):
        sites = [re.sub(r'^==', '', s) for s in m.group(2).strip().splitlines()]
        blocks.append(dict(files=int(m.group(1)), sites=sites))
    return blocks, txt


def run_static(label):
    out = out_dir(label, 'static')
    files = [rel(p) for p in py_files()]
    ruff = json.loads(tool('ruff', 'check', *files, '--select', 'F,B,SIM,PLR,PERF,C90', '--line-length', '120',
                           '--no-fix', '--output-format', 'json') or '[]')
    vult = [ln for ln in tool('vulture', *files, '--min-confidence', '60').splitlines() if ln.strip()]
    cc = json.loads(tool('radon', 'cc', *files, '-s', '-j'))
    raw = json.loads(tool('radon', 'raw', *files, '-j'))
    dups, dup_txt = pylint_duplicates()
    pt = passthrough_scan()
    grades = collections.Counter()
    graded = []
    for f, blocks in cc.items():
        for b in blocks:
            name = b['name'] if b.get('classname') is None else '%s.%s' % (b['classname'], b['name'])
            if b['type'] in ('function', 'method'):
                grades[b['rank']] += 1
                if b['rank'] >= 'C':
                    graded.append(dict(file=f.replace(os.sep, '/'), line=b['lineno'], function=name,
                                       cc=b['complexity'], grade=b['rank'], lines=b['endline'] - b['lineno'] + 1))
    res = dict(
        files={f.replace(os.sep, '/'): dict(loc=v['loc'], sloc=v['sloc'], comments=v['comments'],
                                            multi=v['multi'], blank=v['blank']) for f, v in raw.items()},
        total_loc=sum(v['loc'] for v in raw.values()), total_sloc=sum(v['sloc'] for v in raw.values()),
        n_files=len(files), grades=dict(sorted(grades.items())), grade_c_or_worse=sorted(graded, key=lambda g: -g['cc']),
        ruff_by_rule=dict(collections.Counter(r['code'] for r in ruff).most_common()),
        ruff=[dict(file=rel(r['filename']), line=r['location']['row'], code=r['code'], message=r['message'])
              for r in ruff],
        vulture=vult, duplicates=dups, passthrough=pt)
    with open(os.path.join(out, 'static.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out, 'pylint_duplicate_code.txt'), 'w') as fh:
        fh.write(dup_txt)
    print('STATIC_DONE loc=%d ruff=%d vulture=%d dup=%d passthrough_rows=%d -> %s'
          % (res['total_loc'], len(ruff), len(vult), len(dups), len(pt), out), flush=True)


# coverage part

def run_case(case, label, scratch):
    """One case of the pipeline, Clean's main with the smoke budget (called inside `coverage run`)."""
    import hcommon as H                       # first: thread pins, before torch
    row, noise = case.split('-')
    os.environ['LOCALAPPDATA'] = os.path.join(scratch, 'localappdata-' + case)
    os.makedirs(os.environ['LOCALAPPDATA'], exist_ok=True)
    H.select_side('clean')
    from gantry_dynamic.config import RunConfig, PRESETS
    import gantry_interconnect_dynamic as EP
    cfg = RunConfig(**PRESETS[H.ROWS[int(row)][0]], noise=noise, truncate_s=TRUNCATE_S[noise],
                    out_root=os.path.join(out_dir(label, 'coverage'), 'runs'), **SMOKE)
    EP.main(cfg, run_id='cov-%s-%s' % (label, case))
    H.assert_roots()


def run_coverage(label, scratch, cases):
    out = out_dir(label, 'coverage')
    rc = os.path.join(out, '.coveragerc')
    with open(rc, 'w') as fh:
        fh.write('[run]\nbranch = True\nsource =\n    %s\nomit =\n    %s\n'
                 % (CLEAN, os.path.join(CLEAN, 'blackbox', '*')))
    # per-case data files under another name than the combined one: combine deletes files sharing its prefix
    data_dir = os.path.join(out, 'data')
    os.makedirs(data_dir, exist_ok=True)
    tpath = os.path.join(out, 'case_times.json')
    times = json.load(open(tpath)) if os.path.exists(tpath) else {}
    for case in cases:
        t = time.time()
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        p = subprocess.run((sys.executable, '-u', '-m', 'coverage', 'run', '--rcfile', rc,
                            '--data-file', os.path.join(data_dir, case + '.cov'),
                            os.path.abspath(__file__), '_case', case, '--label', label, '--scratch', scratch),
                           cwd=HERE, env=env)
        times[case] = dict(seconds=round(time.time() - t, 1), rc=p.returncode)
        print('COVERAGE_CASE_END case=%s rc=%d seconds=%.0f' % (case, p.returncode, times[case]['seconds']),
              flush=True)
        with open(tpath, 'w') as fh:
            json.dump(times, fh, indent=1)
    data = [os.path.join(data_dir, c + '.cov') for c in CASES if os.path.exists(os.path.join(data_dir, c + '.cov'))]
    comb = os.path.join(out, '.coverage')
    subprocess.run((sys.executable, '-m', 'coverage', 'combine', '--keep', '--rcfile', rc, '--data-file', comb)
                   + tuple(data), cwd=HERE, check=True)
    subprocess.run((sys.executable, '-m', 'coverage', 'json', '--rcfile', rc, '--data-file', comb,
                    '-o', os.path.join(out, 'combined.json')), cwd=HERE, check=True)
    print('COVERAGE_DONE cases=%d -> %s' % (len(data), out), flush=True)


# report

def innermost(fns, line):
    best = None
    for f in fns:
        if f['start'] <= line <= f['end'] and (best is None or f['start'] >= best['start']):
            best = f
    return best['qual'] if best else '<module>'


def run_report(label):
    st = json.load(open(os.path.join(out_dir(label, 'static'), 'static.json')))
    cov_path = os.path.join(out_dir(label, 'coverage'), 'combined.json')
    unc = []
    totals = None
    if os.path.exists(cov_path):
        cov = json.load(open(cov_path))
        totals = cov['totals']
        grey = {os.path.normcase(p) for p in py_files()}
        for f, d in sorted(cov['files'].items()):
            p = f if os.path.isabs(f) else os.path.join(CLEAN, f)
            if os.path.normcase(os.path.abspath(p)) not in grey:
                continue
            fns = functions(p)[1]
            by_fn = collections.defaultdict(lambda: dict(lines=[], branches=[]))
            for ln in d['missing_lines']:
                by_fn[innermost(fns, ln)]['lines'].append(ln)
            for a, b in d.get('missing_branches', []):
                if a not in d['missing_lines']:     # a branch of a line that never ran is already listed
                    by_fn[innermost(fns, a)]['branches'].append('%d->%s' % (a, b if b > 0 else 'exit'))
            for fn, v in by_fn.items():
                unc.append(dict(file=rel(p), function=fn, lines=v['lines'], branches=v['branches']))
    ops = {}
    for r in PERF_ROWS:
        p = os.path.join(REPO, 'simulations', 'clean-golden', 'perf', label, str(r), 'result.json')
        if os.path.exists(p):
            d = json.load(open(p))
            ops[r] = dict(hfn=d['2a_explain']['hfn']['op_count'],
                          output_only=d['2a_explain']['output_only']['op_count'],
                          step=d.get('2e_step', {}).get('op_count'))
    res = dict(static={k: st[k] for k in ('total_loc', 'total_sloc', 'n_files', 'grades', 'ruff_by_rule')},
               n_ruff=len(st['ruff']), n_vulture=len(st['vulture']), n_duplicates=len(st['duplicates']),
               passthrough_counts=dict(collections.Counter(k.split(' (')[0] for r in st['passthrough']
                                                           for k in r['kinds'])),
               coverage_totals=totals, ops=ops)
    with open(os.path.join(out_dir(label, 'static'), '..', 'scoreboard.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out_dir(label, 'static'), '..', 'findings.tsv'), 'w') as fh:
        fh.write('kind\tsite\tfunction\tdetail\n')
        for u in unc:
            fh.write('uncovered\t%s\t%s\tlines %s; branches %s\n'
                     % (u['file'], u['function'], u['lines'], u['branches']))
        for r in st['passthrough']:
            fh.write('%s\t%s:%d\t%s\tcalls %d, refs %d, outside %s\n'
                     % (', '.join(r['kinds']), r['file'], r['line'], r['function'], r['calls'], r['refs'],
                        r['outside'] or '-'))
        for i, d in enumerate(st['duplicates'], 1):
            fh.write('duplicate\t#%d\t-\t%s\n' % (i, ' | '.join(d['sites'])))
        for g in st['grade_c_or_worse']:
            fh.write('grade %s\t%s:%d\t%s\tCC %d, %d lines\n'
                     % (g['grade'], g['file'], g['line'], g['function'], g['cc'], g['lines']))
    print('REPORT_DONE %s' % json.dumps(res), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=('static', 'coverage', 'report', '_case'))
    ap.add_argument('cases', nargs='*')
    ap.add_argument('--label', required=True)
    ap.add_argument('--scratch', default=None, help='deepSI dumps (coverage); default <out>/coverage/scratch')
    a = ap.parse_args()
    scratch = a.scratch or os.path.join(out_dir(a.label, 'coverage'), 'scratch')
    if a.part == 'static':
        run_static(a.label)
    elif a.part == 'coverage':
        run_coverage(a.label, scratch, a.cases or CASES)
    elif a.part == 'report':
        run_report(a.label)
    else:
        run_case(a.cases[0], a.label, scratch)
