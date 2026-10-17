import os, runpy, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
os.environ['LOCALAPPDATA']=str(root/'.codex-tmp'/'lsq_audit_artifacts')
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import torch
torch.set_num_threads(2)
print('AUDIT_TORCH',torch.__version__,'CUDA_AVAILABLE',torch.cuda.is_available(),flush=True)
runpy.run_path(str(root/'scripts/gantry/thesis-results/test_encoder_lsq.py'),run_name='__main__')
