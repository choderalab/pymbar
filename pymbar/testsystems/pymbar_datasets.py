import os
import numpy as np

try:
    root_dir = os.environ["PYMBAR_DATASETS"]
except KeyError:
    root_dir = os.environ["HOME"]

def get_sn(N_k):
    """MAY HAVE EMPTY STATES"""
    n_states = len(N_k)
    s_n = np.zeros(sum(N_k), 'int')
    k = 0
    for i in range(n_states):
        for n in range(N_k[i]):
            s_n[k] = i
            k += 1
    return s_n

def load_from_hdf(filename):
    import tables
    f = tables.File(filename, 'r')
    u_kn = f.root.u_kn[:]
    N_k = f.root.N_k[:]
    f.close()
    s_n = get_sn(N_k)
    return u_kn, N_k, s_n

def load_gas_data():
    name = "gas-properties"
    u_kn, N_k, s_n = load_from_hdf(os.path.join(root_dir, name, "%s.h5" % name))
    return name, u_kn, N_k, s_n

def load_8proteins_data():
    name = "8proteins"
    u_kn, N_k, s_n = load_from_hdf(os.path.join(root_dir, name, "%s.h5" % name))
    return name, u_kn, N_k, s_n
