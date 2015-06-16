import os
import numpy as np
from pymbar.utils import ensure_type
from os import environ
from os.path import join
from os.path import exists
from os.path import expanduser
from os import makedirs

def get_data_home(data_home=None):
    """Return the path of the pymbar data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'pymbar_data'
    in the user home folder.

    Alternatively, it can be set by the 'PYMBAR_DATA' environment
    variable or programmatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = environ.get('PYMBAR_DATA', join('~', 'pymbar_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home

def get_sn(N_k):
    """Assuming the usual ordering of samples and states, guess the
    the state origin of each sample.
    
    Parameters
    ----------
    N_k : np.ndarray, dtype='int', shape=(n_states)
        The number of samples from each state.

    Returns
    -------
    s_n : np.ndarray, dtype=int, shape=(n_samples)
        The (guessed) state of origin of each state.  
    
    Notes
    -----
    The output MAY HAVE EMPTY STATES.
    """
    n_states = len(N_k)
    s_n = np.zeros(sum(N_k), 'int')
    k = 0
    for i in range(n_states):
        for n in range(N_k[i]):
            s_n[k] = i
            k += 1
    return s_n

def load_from_hdf(filename):
    """Load an HDF5 file that was created via save().
    Parameters
    ----------
    filename : str
        filename of HDF5
    
    Returns
    -------

    u_kn : np.ndarray, dtype='float', shape=(n_states, n_samples)
        Reduced potential energies
    N_k : np.ndarray, dtype='int', shape=(n_states)
        Number of samples taken from each state
    s_n : np.ndarray, optional, default=None, dtype=int, shape=(n_samples)
        The state of origin of each state.  If none, guess the state origins.
    
    """
    import tables
    try:
        f = tables.File(filename, 'r')
    except IOError as e:
        print("Cannot load %s.  Please download pymbar-datasets and export PYMBAR_DATASETS to point to its location." % filename)
        raise(e)

    u_kn = f.root.u_kn[:]
    N_k = f.root.N_k[:]
    s_n = f.root.s_n[:]
    f.close()
    return u_kn, N_k, s_n

def load_gas_data():
    name = "gas-properties"
    u_kn, N_k, s_n = load_from_hdf(os.path.join(get_data_home(), name, "%s.h5" % name))
    return name, u_kn, N_k, s_n

def load_8proteins_data():
    name = "8proteins"
    u_kn, N_k, s_n = load_from_hdf(os.path.join(get_data_home(), name, "%s.h5" % name))
    return name, u_kn, N_k, s_n

def load_k69_data():
    name = "k69"
    u_kn, N_k, s_n = load_from_hdf(os.path.join(get_data_home(), name, "%s.h5" % name))
    return name, u_kn, N_k, s_n



def save(name, u_kn, N_k, s_n=None, least_significant_digit=None):
    """Create an HDF5 dump of an existing MBAR job for later use / testing.
    
    Parameters
    ----------
    name : str
        Name of dataset
    u_kn : np.ndarray, dtype='float', shape=(n_states, n_samples)
        Reduced potential energies
    N_k : np.ndarray, dtype='int', shape=(n_states)
        Number of samples taken from each state
    s_n : np.ndarray, optional, default=None, dtype=int, shape=(n_samples)
        The state of origin of each state.  If none, guess the state origins.
    least_significant_digit : int, optional, default=None
        If not None, perform lossy compression using tables.Filter(least_significant_digit=least_significant_digit)

    Notes
    -----
    The output HDF5 files should be readible by the helper funtions pymbar_datasets.py
    """
    import tables
    
    (n_states, n_samples) = u_kn.shape
    
    u_kn = ensure_type(u_kn, 'float', 2, "u_kn or Q_kn", shape=(n_states, n_samples))
    N_k = ensure_type(N_k, 'int64', 1, "N_k", shape=(n_states,))

    if s_n is None:
        s_n = get_sn(N_k)

    s_n = ensure_type(s_n, 'int64', 1, "s_n", shape=(n_samples,))

    hdf_filename = os.path.join("./", "%s.h5" % name)
    f = tables.File(hdf_filename, 'a')
    f.createCArray("/", "u_kn", tables.Float64Atom(), obj=u_kn, filters=tables.Filters(complevel=9, complib="zlib", least_significant_digit=least_significant_digit))
    f.createCArray("/", "N_k", tables.Int64Atom(), obj=N_k, filters=tables.Filters(complevel=9, complib="zlib"))
    f.createCArray("/", "s_n", tables.Int64Atom(), obj=s_n, filters=tables.Filters(complevel=9, complib="zlib"))
    f.close()
