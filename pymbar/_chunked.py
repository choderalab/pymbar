"""Chunked working-array variants of the MBAR inner-fit operations.

The default `mbar_solvers.py` calls

    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)

which materializes a (N, K) working array the same size as ``u_kn``. For
large N, this is the dominant peak-memory cost during the fit -- not
``u_kn`` itself. See pymbar issue #574.

These helpers compute the same quantities by streaming over the sample
axis, with peak working-array memory ``O(chunk_size * K)`` instead of
``O(N * K)``. Numpy-only; the JAX path is preserved for callers that fit
in memory.
"""
import numpy as np


def chunked_log_denominator(u_kn, N_k, f_k, chunk_size):
    """Streaming logsumexp(f_k - u_kn.T, b=N_k, axis=1).

    log_denom[n] = log sum_k N_k * exp(f_k - u_kn[k, n])
    """
    N = u_kn.shape[1]
    with np.errstate(divide='ignore'):
        log_N = np.log(N_k).astype(u_kn.dtype)  # -inf for N_k = 0; drops cleanly
    a = (f_k + log_N).astype(u_kn.dtype)  # (K,)
    out = np.empty(N, dtype=u_kn.dtype)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        z = a[:, None] - u_kn[:, s:e]  # (K, B)
        m = z.max(axis=0)
        np.subtract(z, m, out=z)
        np.exp(z, out=z)
        out[s:e] = m + np.log(z.sum(axis=0))
    return out


def chunked_log_numerator_k(u_kn, log_denominator_n, chunk_size):
    """Streaming logsumexp(-log_denominator_n - u_kn, axis=1).

    log_num_k[k] = log sum_n exp(-log_denominator_n[n] - u_kn[k, n])

    Maintains a running max-shift per state across chunks.
    """
    K, N = u_kn.shape
    m_k = np.full(K, -np.inf, dtype=u_kn.dtype)
    S_k = np.zeros(K, dtype=u_kn.dtype)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        z = -log_denominator_n[s:e][None, :] - u_kn[:, s:e]  # (K, B)
        m_chunk = z.max(axis=1)
        m_new = np.maximum(m_k, m_chunk)
        # exp(m_k - m_new) is 0 when m_k = -inf (initial); guard the all-inf
        # corner case where m_new is also -inf.
        with np.errstate(invalid='ignore'):
            scale = np.where(np.isfinite(m_new), np.exp(m_k - m_new), 0.0)
        np.subtract(z, m_new[:, None], out=z)
        np.exp(z, out=z)
        S_k = S_k * scale + z.sum(axis=1)
        m_k = m_new
    return m_k + np.log(S_k)


def chunked_mbar_objective(u_kn, N_k, f_k, chunk_size):
    """Equivalent to jax_mbar_objective."""
    log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    return float(np.sum(log_denom_n) - np.dot(N_k, f_k))


def chunked_mbar_gradient(u_kn, N_k, f_k, chunk_size):
    """Equivalent to jax_mbar_gradient."""
    log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    log_num_k = chunked_log_numerator_k(u_kn, log_denom_n, chunk_size)
    return -1.0 * N_k * (1.0 - np.exp(f_k + log_num_k))


def chunked_mbar_objective_and_gradient(u_kn, N_k, f_k, chunk_size):
    """Equivalent to jax_mbar_objective_and_gradient (single log_denom pass)."""
    log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    obj = float(np.sum(log_denom_n) - np.dot(N_k, f_k))
    log_num_k = chunked_log_numerator_k(u_kn, log_denom_n, chunk_size)
    grad = -1.0 * N_k * (1.0 - np.exp(f_k + log_num_k))
    return obj, grad


def chunked_mbar_hessian(u_kn, N_k, f_k, chunk_size):
    """Equivalent to jax_mbar_hessian. Streams W_chunk = exp(...) of shape
    (B, K) and accumulates H += W_chunk.T @ W_chunk into a (K, K) buffer."""
    log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    K, N = u_kn.shape
    H = np.zeros((K, K), dtype=u_kn.dtype)
    W_sum = np.zeros(K, dtype=u_kn.dtype)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        W = np.exp(f_k[None, :] - u_kn[:, s:e].T - log_denom_n[s:e, None])
        H += W.T @ W
        W_sum += W.sum(axis=0)
    H *= N_k
    H *= N_k[:, None]
    H -= np.diag(W_sum * N_k)
    return -H


def chunked_self_consistent_update(u_kn, N_k, f_k, chunk_size):
    """Equivalent to _jit_self_consistent_update."""
    log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    return -chunked_log_numerator_k(u_kn, log_denom_n, chunk_size)


def chunked_precondition_u_kn(u_kn, N_k, f_k, chunk_size):
    """In-place precondition: ``u_kn -= u_kn.min(0); u_kn += log_denom - shift``.

    Mutates ``u_kn``. Caller must rebind, like ``u_kn = precondition_u_kn(u_kn, ...)``,
    to match jax_precondition_u_kn semantics. MBAR is invariant under per-sample
    shifts so f_k and Log_W_nk are unaffected.
    """
    u_min = u_kn.min(axis=0)  # (N,)
    np.subtract(u_kn, u_min, out=u_kn)
    log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    shift = log_denom_n - float(np.dot(N_k, f_k) / N_k.sum())
    np.add(u_kn, shift, out=u_kn)
    return u_kn


def chunked_mbar_log_W_nk(u_kn, N_k, f_k, chunk_size):
    """Equivalent to jax_mbar_log_W_nk. The output is shape (N, K) so this
    only chunks the *peak* during construction, not the result.
    """
    log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    K, N = u_kn.shape
    out = np.empty((N, K), dtype=u_kn.dtype)
    for s in range(0, N, chunk_size):
        out[s:s + chunk_size] = (
            f_k[None, :] - u_kn[:, s:s + chunk_size].T
            - log_denom_n[s:s + chunk_size, None])
    return out
