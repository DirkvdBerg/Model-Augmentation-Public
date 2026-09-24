p='audit_sensitivity.py'
s=open(p,encoding='utf-8').read()
s=s.replace("""    res = {}
    for g, files in GROUPS.items():""","""    out_path = os.path.join(HERE, 'outputs', 'audit_sensitivity.json')
    prev = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res = prev.get('groups', {})
    sel = [g for g in os.environ.get('AUDIT_GROUPS', ','.join(GROUPS)).split(',') if g]
    for g, files in ((g, GROUPS[g]) for g in sel):""")
s=s.replace("""        del J, b
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
""","""        del J, b
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        with open(out_path, 'w') as fh:           # incremental: a kill loses one group at most
            json.dump(dict(prev, mode=MODE, stride=stride, combo=list(COMBO), groups=res), fh,
                      indent=1)
    if os.environ.get('AUDIT_COVERAGE', '1') != '1':
        return
    cstride = int(os.environ.get('AUDIT_COV_STRIDE', str(stride)))
""")
s=s.replace("""    Z = lambda fl: build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, files=fl, stride=stride,  # noqa: E731""","""    Z = lambda fl: build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, files=fl, stride=cstride,  # noqa: E731""")
s=s.replace("""    with open(os.path.join(HERE, 'outputs', 'audit_sensitivity.json'), 'w') as fh:
        json.dump(dict(mode=MODE, stride=stride, combo=list(COMBO), groups=res,
                       coverage=dict(h_train_median_nn=h, per_record=cov)), fh, indent=1)""","""    with open(out_path, 'w') as fh:
        json.dump(dict(mode=MODE, stride=stride, combo=list(COMBO), groups=res,
                       coverage=dict(h_train_median_nn=h, per_record=cov, stride=cstride)), fh,
                  indent=1)""")
open(p,'w',encoding='utf-8').write(s)
print(s.count('AUDIT_GROUPS'), s.count('cstride'))
