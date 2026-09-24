import json
d=json.load(open('outputs/audit_sensitivity.json'))
C=d['combo']
print('group n rank cond gamma cos rho linRMS', ' '.join(C))
for g,r in d['groups'].items():
    print('%-18s %6d %2d %9.3e %7.2f %+.3f %.4f %8.1f | %s'%(g,r['n_tuples'],r['rank'],r['cond'],r['gamma'],r['cos_cg1_cg2'],r['rho'],r['bias_lin_rms'],' '.join('%+.0f'%r['bias_lin_pct'][c] for c in C)))
print('sigma_min/row', {g:'%.2e'%r['sigma_min_per_row'] for g,r in d['groups'].items()})
print('weak', {g:r['weakest_direction'] for g,r in d['groups'].items()})
print('cov h', d['coverage']['h_train_median_nn'])
