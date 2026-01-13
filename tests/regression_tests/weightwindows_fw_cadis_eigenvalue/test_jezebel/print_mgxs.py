import h5py
import numpy as np

f = h5py.File('mgxs.h5', 'r')

for mat_name, mat in f.items():
    m = mat['294K']
    for xs_name, xs in m.items():
        print(f'{mat_name} {xs_name}: {np.array(xs)}')
