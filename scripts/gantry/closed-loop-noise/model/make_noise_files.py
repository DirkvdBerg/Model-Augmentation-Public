"""G5: per-record encoder-node noise v for the MATLAB generator copies (CN-009).

    python model/make_noise_files.py [reading=tanh] [scale=1.0] [tag=ss]
For every record of the production dataset, v (T x 3, stage [X1, X2, Y], metres) from H_v
(outputs/g3_source/hv_<reading>.npz), T = the record's own length, seed = crc32(record id) +
offset(tag). Writes matlab/noise_<tag>/<id>.mat (v, seed, reading, scale, hv_sha256) and a
manifest. The generator adds v at the encoder node; v = 0 is the noise-off mode.
"""
import hashlib
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402
from scipy.io import loadmat, savemat, whosmat                          # noqa: E402

from model import noise_model as nm                                     # noqa: E402

READING = sys.argv[1] if len(sys.argv) > 1 else 'tanh'
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
TAG = sys.argv[3] if len(sys.argv) > 3 else 'ss'
PROD = os.path.join(cn_env.REPO, 'data', 'gantry', 'matlab', 'trajectory',
                    'augmentation_ma50_z03_b140-230_a6all_telica_coulomb')
HV = os.path.join(cn_env.OUTPUTS, 'g3_source', f'hv_{READING}.npz')
OUTD = os.path.join(cn_env.CN, 'matlab', f'noise_{TAG}')
SEED_OFFSET = {'ss': 0, 'motion': 7919}


def main():
    os.makedirs(OUTD, exist_ok=True)
    hv = np.load(HV)
    A, Sig = hv['A'], hv['Sig']
    sha = hashlib.sha256(open(HV, 'rb').read()).hexdigest()
    Psi = nm.impulse_fft(A)
    ids = sorted(f[:-4] for f in os.listdir(PROD) if f.endswith('.mat'))
    man = dict(reading=READING, scale=SCALE, tag=TAG, hv=os.path.relpath(HV, cn_env.CN), hv_sha256=sha,
               order=int(A.shape[0]), impulse_lags=len(Psi), records={})
    for rid in ids:
        info = {n: s for n, s, _ in whosmat(os.path.join(PROD, rid + '.mat'))}
        T = int(info['y'][0])
        seed = (zlib.crc32(rid.encode()) + SEED_OFFSET[TAG]) % (2 ** 32)
        v = SCALE * nm.generate_fft(A, Sig, T, np.random.default_rng(seed), Psi=Psi)
        savemat(os.path.join(OUTD, rid + '.mat'),
                dict(v=v, seed=seed, reading=READING, scale=SCALE, hv_sha256=sha, fs=float(hv['fs'])))
        man['records'][rid] = dict(T=T, seed=int(seed), rms_nm=(v.std(0) * 1e9).round(3).tolist())
        print(f'[noise] {rid}: T {T}, seed {seed}, rms {np.round(v.std(0) * 1e9, 2)} nm')
    json.dump(man, open(os.path.join(OUTD, 'manifest.json'), 'w'), indent=1)
    print(f'[noise] {len(ids)} files in {os.path.relpath(OUTD, cn_env.CN)}')


if __name__ == '__main__':
    main()
