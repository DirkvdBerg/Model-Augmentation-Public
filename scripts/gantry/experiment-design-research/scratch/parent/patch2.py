p='audit_sensitivity.py'
s=open(p,encoding='utf-8').read()
s=s.replace("""from gantry_dynamic.data import compute_normalization, load_datasets               # noqa: E402""",
"""from gantry_dynamic.data import TRAIN_FILES                                         # noqa: E402""")
s=s.replace("""def collinearity(J):""","""def light_norm(cfg):
    \"\"\"The four statistics `build_reference_set` and the block read from `Norm`, computed with
    the formulas of `data.py::compute_normalization` (lines 214-251) record by record, so the
    29-record `load_datasets` (about 1 GB resident, which tripped the watchdog's RAM guard with
    about 3.5 GB free on the host) is not needed. float32 cast as the pipeline's train_list.\"\"\"
    PinvT = np.linalg.inv(P_pt.detach().cpu().numpy().T).astype(np.float32)
    xs, us = [], []
    for f in TRAIN_FILES:
        u, y, _xl, _ = _load_record_float64(f, cfg)
        u, y = u.astype(np.float32), y.astype(np.float32)
        pos = (PinvT @ y.T).T
        vel = np.diff(pos, axis=0) / np.float32(cfg.ts_new)
        vel = np.vstack([vel[:1], vel])
        xs.append(np.hstack([pos, vel]))
        us.append(u)
    x_all, u_all = np.concatenate(xs), np.concatenate(us)
    from types import SimpleNamespace
    return SimpleNamespace(
        x_mean=x_all.mean(0).reshape(6, 1).astype(np.float32),
        std_x=x_all.std(0).reshape(6, 1).astype(np.float32) + 1e-8,
        std_u=u_all.std(0).reshape(3, 1).astype(np.float32) + 1e-8,
        u_mean=u_all.mean(0).reshape(3, 1).astype(np.float32))


def collinearity(J):""")
s=s.replace("""    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)""","""    norm = light_norm(cfg)
    print('[norm] std_x', norm.std_x.ravel(), 'std_u', norm.std_u.ravel(), flush=True)""")
open(p,'w',encoding='utf-8').write(s)
