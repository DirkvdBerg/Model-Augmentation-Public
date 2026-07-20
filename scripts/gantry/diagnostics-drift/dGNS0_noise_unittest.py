"""
dGNS0_noise_unittest.py -- GNS diagnostic, STAGE 0: unit-test the random-walk noise injector in ISOLATION,
before any training. Plan: docs/gns-encoder-diagnostic-plan.md (Stage 0).

Why this exists: the GNS-with-encoder diagnostic (docs/rollout-stability-literature.md GNS-fit section)
injects GNS-style random-walk noise onto selected state rows during the rollout. Before wiring it into the
pipeline we verify the GENERATOR in isolation (diagnostic-must-not-require-the-thing-it-verifies): it must
(1) produce a correct random walk (Var[n_k] grows LINEARLY as k*sigma^2, zero-mean increments),
(2) inject ONLY on the intended rows and leave every other row -- especially the Y-position (scheduling)
    row -- BIT-IDENTICAL,
(3) NOT modify the clean target array,
(4) be reproducible under a fixed seed.

The injector is deliberately INDEX-AGNOSTIC (takes explicit `rows`) so the mechanism is verified independent
of the state-ordering convention; the concrete pipeline row indices (and confirming the Y-position row) are
resolved in Stage 1 against the actual pipeline state. NOTE: two state conventions exist in the repo --
truth/drift_common `[X, Th, Y, da, dX, dTh, dY, vda]` vs the pipeline model state; Y-POSITION is index 2 in
the truth convention. Stage 1 must confirm the pipeline's Y-position index before injecting.

No training, no pipeline, no framework. Pure numpy. Runs in milliseconds.

Run:
  PYTHONIOENCODING=utf-8 conda run -n GraduationProject python \
     scripts/gantry/diagnostics-drift/dGNS0_noise_unittest.py
"""
import os
import json

import numpy as np

OUT_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')),
    'simulations', 'gantry_subnet', 'diagnostics',
)
os.makedirs(OUT_DIR, exist_ok=True)


# ── The injector under test ──────────────────────────────────────────────────
def random_walk(n_steps, dim, sigma, rng):
    """Random-walk sequence n_k = sum_{i<=k} eps_i, eps_i ~ N(0, sigma^2), shape (n_steps, dim).

    GNS (2002.09405): accumulated (integration-like) noise, NOT i.i.d. per step. Var[n_k] = k*sigma^2.
    """
    increments = rng.normal(0.0, sigma, size=(n_steps, dim))
    return np.cumsum(increments, axis=0)


def inject(state_seq, rows, sigma, rng):
    """Return a COPY of state_seq (shape (T, nx)) with random-walk noise added ONLY on `rows`.

    Every column not in `rows` is left bit-identical. The input array is not mutated.
    """
    out = state_seq.copy()
    rows = list(rows)
    if rows:
        walk = random_walk(state_seq.shape[0], len(rows), sigma, rng)  # (T, len(rows))
        out[:, rows] = out[:, rows] + walk
    return out


# ── Checks ───────────────────────────────────────────────────────────────────
def check_randomwalk_statistics():
    """(1) Var[n_k] ~ k*sigma^2 (linear), increments zero-mean. Averaged over many realizations."""
    rng = np.random.default_rng(0)
    T, R, sigma = 400, 2, 3.0
    n_real = 20000
    walks = np.stack([random_walk(T, R, sigma, rng) for _ in range(n_real)], axis=0)  # (n_real, T, R)
    emp_var = walks.var(axis=0).mean(axis=1)                       # (T,) empirical Var[n_k]
    pred_var = (np.arange(1, T + 1)) * sigma ** 2                  # k*sigma^2
    rel_err = np.abs(emp_var - pred_var) / pred_var
    # increments zero-mean
    incr = np.diff(walks, axis=1)
    incr_mean = np.abs(incr.mean())
    passed = bool(rel_err.max() < 0.05 and incr_mean < 0.05 * sigma)
    return passed, {
        'max_rel_var_err': float(rel_err.max()),
        'incr_abs_mean_over_sigma': float(incr_mean / sigma),
        'emp_var_step': emp_var.tolist(),
        'pred_var_step': pred_var.tolist(),
    }


def check_row_isolation():
    """(2) Only `rows` change; all others -- especially the Y-position row -- bit-identical."""
    rng = np.random.default_rng(1)
    T, nx = 500, 8
    Y_POS_ROW = 2                       # scheduling variable; must NEVER be perturbed
    state = rng.normal(0.0, 1.0, size=(T, nx))
    rows = [0, 3, 5]                    # example X-pos + two velocity rows (index-agnostic test)
    assert Y_POS_ROW not in rows, "test setup error: Y-position must be excluded"
    out = inject(state, rows, sigma=2.0, rng=rng)
    changed = [j for j in range(nx) if not np.array_equal(out[:, j], state[:, j])]
    untouched_ok = all(np.array_equal(out[:, j], state[:, j]) for j in range(nx) if j not in rows)
    y_pos_identical = np.array_equal(out[:, Y_POS_ROW], state[:, Y_POS_ROW])
    passed = bool(sorted(changed) == sorted(rows) and untouched_ok and y_pos_identical)
    return passed, {
        'rows_injected': rows,
        'rows_changed': sorted(changed),
        'y_pos_row': Y_POS_ROW,
        'y_pos_bit_identical': bool(y_pos_identical),
        'nontarget_rows_bit_identical': bool(untouched_ok),
    }


def check_target_unmodified():
    """(3) The clean target array is not modified (input not mutated; target is a separate object)."""
    rng = np.random.default_rng(2)
    T, nx = 300, 8
    state = rng.normal(size=(T, nx))
    target = state.copy()               # a separate "clean truth" array
    state_before = state.copy()
    _ = inject(state, rows=[0, 3, 5], sigma=1.5, rng=rng)
    input_untouched = np.array_equal(state, state_before)
    target_untouched = np.array_equal(target, state_before)
    passed = bool(input_untouched and target_untouched)
    return passed, {'input_not_mutated': bool(input_untouched), 'target_not_mutated': bool(target_untouched)}


def check_reproducibility():
    """(4) Same seed -> identical output; different seed -> different output."""
    T, nx = 200, 8
    state = np.random.default_rng(9).normal(size=(T, nx))
    a = inject(state, [0, 3, 5], 2.0, np.random.default_rng(42))
    b = inject(state, [0, 3, 5], 2.0, np.random.default_rng(42))
    c = inject(state, [0, 3, 5], 2.0, np.random.default_rng(43))
    same = np.array_equal(a, b)
    diff = not np.array_equal(a, c)
    passed = bool(same and diff)
    return passed, {'same_seed_identical': bool(same), 'diff_seed_differs': bool(diff)}


def _falsifiable_plot(stats):
    """Var[n_k] vs step: predicted k*sigma^2 line vs empirical points. Poses the test, does not assert it."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        return None
    emp = np.array(stats['statistics'][1]['emp_var_step'])
    pred = np.array(stats['statistics'][1]['pred_var_step'])
    k = np.arange(1, len(emp) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k, pred, 'k-', label='predicted  k*sigma^2')
    ax.plot(k[::10], emp[::10], 'r.', ms=4, label='empirical Var[n_k]')
    ax.set_xlabel('step k'); ax.set_ylabel('Var[n_k]')
    ax.set_title('Stage 0: is the injector a correct random walk? (Var linear in k?)')
    ax.legend()
    path = os.path.join(OUT_DIR, 'dGNS0_randomwalk_variance.png')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path


def main():
    results = {}
    for name, fn in [
        ('statistics', check_randomwalk_statistics),
        ('row_isolation', check_row_isolation),
        ('target_unmodified', check_target_unmodified),
        ('reproducibility', check_reproducibility),
    ]:
        passed, detail = fn()
        results[name] = (passed, detail)
        flag = 'PASS' if passed else 'FAIL'
        print(f'[{flag}] {name}')
        for k, v in detail.items():
            if isinstance(v, list):
                continue
            print(f'         {k} = {v}')

    all_pass = all(p for p, _ in results.values())
    plot_path = _falsifiable_plot(results)

    out = {
        'stage': 0,
        'all_pass': bool(all_pass),
        'checks': {k: {'pass': bool(p), **{kk: vv for kk, vv in d.items() if not isinstance(vv, list)}}
                   for k, (p, d) in results.items()},
        'plot': plot_path,
    }
    json_path = os.path.join(OUT_DIR, 'dGNS0_noise_unittest.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)

    print()
    print(f'ALL PASS: {all_pass}')
    print(f'json -> {json_path}')
    if plot_path:
        print(f'plot -> {plot_path}')
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
