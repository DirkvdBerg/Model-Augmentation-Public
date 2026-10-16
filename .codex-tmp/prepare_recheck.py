from pathlib import Path
p=Path('.codex-tmp/audit_final_run_20261016.py')
s=p.read_text()
s=s.replace('import ast\n','import ast\nimport json\n')
s=s.replace('servo_sigma=servo_sigma, window_controller_index=window_controller_index,','servo_sigma=servo_sigma, window_controller_index=window_controller_index, json=json,')
s=s.replace('model.set_objective([sd.y for sd in records], 3, ry, (history, failure))','hist_path = SCRATCH / "recheck_bb_history.json"\nmodel.set_objective([sd.y for sd in records], 3, ry, (history, failure), str(hist_path))')
a=s.index("print('BB_EVALUATOR_NAME'")
b=s.index("ev = ROOT",a)
s=s[:a]+'''saved_history = json.loads(hist_path.read_text())
assert len(saved_history['history']) == len(history) >= 3
assert not failure and saved_history['failure'] is None
assert (SCRATCH / 'deepSI/checkpoints' / (model.name + '_best.pth')).exists()
print('BB_CHECKPOINT_AND_DURABLE_HISTORY_OK', len(history), flush=True)

'''+s[b:]
a=s.index("print('CAMPAIGN_CACHE_COLLISION'")
b=s.index('import gantry_dynamic.obc_gantry',a)
s=s[:a]+'''assert old['CACHE'] != new['CACHE']
old['check_identity']('R1_val', {'job': np.array(old['RUNS']['R1'])})
try:
    old['check_identity']('R1_val', {'job': np.array(-1)})
except ValueError:
    pass
else:
    raise AssertionError('mismatched cache accepted')
print('CAMPAIGN_CACHE_AND_IDENTITY_OK', flush=True)
for name in ('eval_greybox.py', 'eval_overlap.py', 'eval_negation.py'):
    source = (ev.parent / name).read_text(encoding='utf-8-sig')
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == '_sync_prediction_state_for_validation'
                   for n in ast.walk(ast.parse(source)))
print('INFERENCE_NO_BASIS_REBUILD_OK', flush=True)

'''+s[b:]
a=s.index("same = ((ref.record_ix")
b=s.index('class Life:',a)
s=s[:a]+'''source = ast.parse((ev.parent / 'eval_overlap.py').read_text(encoding='utf-8-sig'))
branch = next(n for n in ast.walk(source) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Attribute)
              and n.test.left.attr == 'x_phys_next_rec')
f_true = torch.zeros((ref.n,6), dtype=torch.float64)
Js = torch.arange(ref.n*6*2, dtype=torch.float64).reshape(ref.n*6,2)
local = dict(torch=torch, np=np, ref=ref, norm=norm, row_ix=list(range(6)), f_true=f_true, Js=Js)
exec(compile(ast.Module(body=branch.body, type_ignores=[]), '<bias-stack>', 'exec'), local)
assert local['Phi'].shape[0] == local['xi'].shape[0] == ref.n*6
assert torch.equal(local['xi'], ref.x_phys_next_rec.reshape(-1))
print('COMPLETE_NORMALIZED_BIAS_STACK_OK', ref.n, 'tuples', flush=True)

'''+s[b:]
a=s.index('# Exercise the real preflight')
s=s[:a]+'''# Exercise actual helper plumbing with mocked data/model, without a production rollout.
import gantry_dynamic.data as gd_data
import gantry_dynamic.model as gd_model
import gantry_dynamic.controller as gd_ctrl
helpers = runpy.run_path(str(ROOT / 'scripts/gantry/thesis-results/test_ps2_pipeline.py'))
fake_data = SimpleNamespace(train_files=['train'], val_files=['val'], val_ckpt_data=[],
                            e_train=[np.ones((40,3))], e_val=[np.ones((40,3))*2])
fake_norm = SimpleNamespace(ystd=np.ones(3), std_u=np.ones(3))
fake_fs = SimpleNamespace(na=30, nb=30)
with patch.object(gd_data, 'load_datasets', return_value=fake_data), \\
     patch.object(gd_data, 'compute_normalization', return_value=fake_norm), \\
     patch.object(gd_model, 'build_model', return_value=fake_fs), \\
     patch.object(gd_ctrl, 'build_closed_loop', return_value=SimpleNamespace()) as closed:
    helpers['build']('u')
    assert closed.call_args.kwargs['e_train'] is fake_data.e_train
    assert closed.call_args.kwargs['e_val'] is fake_data.e_val
print('PREFLIGHT_SERVO_ERROR_PLUMBING_OK', flush=True)
print('RECHECK_PROBES_COMPLETE', flush=True)
'''
Path('.codex-tmp/recheck_final_run_20261016.py').write_text(s)
