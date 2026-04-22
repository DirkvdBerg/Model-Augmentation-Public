import torch
d = torch.load('simulations/param_recovery/segment_len_diag_T1-T6-T2-T3-T4-T5_v1.pt', weights_only=False)
print("chosen_segment_len:", d['chosen_segment_len'])
print("candidates:", d['candidate_lengths_samples'])
print("candidate times (s):", d.get('candidate_lengths_s', 'N/A'))
for r in d.get('results', []):
    print(f"  seg={r['segment_len']:>5}  rmse={r.get('rmse_m', r.get('rmse', '?'))}")
