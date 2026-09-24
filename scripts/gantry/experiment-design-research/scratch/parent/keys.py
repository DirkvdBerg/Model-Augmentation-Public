import sys, numpy as np
from scipy.io import loadmat, whosmat
for f in sys.argv[1:]:
    print(f)
    for n,s,c in whosmat(f): print('   ',n,s,c)
