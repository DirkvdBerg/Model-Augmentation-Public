"""Quick smoke test: does build_model with ENCODER_INIT='linear_map' work?"""
import sys, os
# Change to gantry script dir so relative paths resolve correctly
GANTRY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(GANTRY_DIR)
sys.path.insert(0, os.path.join(GANTRY_DIR, '..', '..'))

# Execute the training script up to (but not including) train_model
script_path = os.path.join(GANTRY_DIR, 'gantry_interconnect_dynamic.py')
with open(script_path) as f:
    code = f.read()
# Execute everything before train_model definition
code_setup = code.split('def train_model')[0]
ns = {'__file__': script_path, '__name__': '__main__', '__builtins__': __builtins__}
exec(compile(code_setup, script_path, 'exec'), ns)

hp = ns['DEFAULT_HP'].copy()
fit_sys = ns['build_model'](hp)
print('build_model OK')
print(f'Encoder type: {type(fit_sys.encoder).__name__}')
print(f'Encoder params: {sum(p.numel() for p in fit_sys.encoder.parameters())}')
for name, p in fit_sys.encoder.named_parameters():
    print(f'  {name}: {p.shape}')

# Quick test: forward pass through encoder with dummy data
import torch
batch = 4
# With na_right=1, nb_right=1, the encoder sees na+1 and nb+1 time steps
na_total = 25 + 1  # na + na_right
nb_total = 25 + 1
uhist = torch.randn(batch, nb_total, 3, dtype=torch.float32)
yhist = torch.randn(batch, na_total, 3, dtype=torch.float32)
x0 = fit_sys.encoder(uhist, yhist)
print(f'\nEncoder output shape: {x0.shape}  (expect ({batch}, {6 + hp["NX_ANN"]}))')
print('PASS')
