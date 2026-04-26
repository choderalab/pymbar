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

The helpers accept either a dense ``(K, N)`` ndarray or a zero-arg
callable yielding ``(K, B)`` chunks (so even ``u_kn`` itself can be
streamed from disk).
"""
import numpy as np


def _iter_chunks(u_kn, chunk_size):
    """Yield (start_index, (K, B) chunk) pairs from either dense u_kn or a
    callable. The callable form must return a fresh iterator on each call
    (so it can be re-iterated within a fit) and must yield deterministic
    chunks summing to the full N.
    """
    if isinstance(u_kn, np.ndarray):
        N = u_kn.shape[1]
        for s in range(0, N, chunk_size):
            yield s, u_kn[:, s:s + chunk_size]
    elif callable(u_kn):
        s = 0
        for chunk in u_kn():
            yield s, chunk
            s += chunk.shape[1]
    else:
        raise TypeError(
            f"u_kn must be ndarray or zero-arg callable, got {type(u_kn).__name__}")


def _shape_of(u_kn, N_k):
    """Return (K, N) for either dense ndarray or callable u_kn. For callables,
    K and N are inferred from N_k.
    """
    if isinstance(u_kn, np.ndarray):
        return u_kn.shape
    return len(N_k), int(np.sum(N_k))


def _dtype_of(u_kn):
    """Best-effort dtype detection; falls back to float64 for callables."""
    return u_kn.dtype if isinstance(u_kn, np.ndarray) else np.float64


def chunked_log_denominator(u_kn, N_k, f_k, chunk_size):
    """Streaming logsumexp(f_k - u_kn.T, b=N_k, axis=1).

    log_denom[n] = log sum_k N_k * exp(f_k - u_kn[k, n])
    """
    _, N = _shape_of(u_kn, N_k)
    dtype = _dtype_of(u_kn)
    with np.errstate(divide='ignore'):
        log_N = np.log(N_k).astype(dtype)  # -inf for N_k = 0; drops cleanly
    a = (f_k + log_N).astype(dtype)  # (K,)
    out = np.empty(N, dtype=dtype)
    for s, chunk in _iter_chunks(u_kn, chunk_size):
        e = s + chunk.shape[1]
        z = a[:, None] - chunk  # (K, B)
        m = z.max(axis=0)
        np.subtract(z, m, out=z)
        np.exp(z, out=z)
        out[s:e] = m + np.log(z.sum(axis=0))
    return out


def chunked_log_numerator_k(u_kn, N_k, log_denominator_n, chunk_size):
    """Streaming logsumexp(-log_denominator_n - u_kn, axis=1).

    log_num_k[k] = log sum_n exp(-log_denominator_n[n] - u_kn[k, n])

    Maintains a running max-shift per state across chunks.
    """
    K, _ = _shape_of(u_kn, N_k)
    dtype = _dtype_of(u_kn)
    m_k = np.full(K, -np.inf, dtype=dtype)
    S_k = np.zeros(K, dtype=dtype)
    for s, chunk in _iter_chunks(u_kn, chunk_size):
        e = s + chunk.shape[1]
        z = -log_denominator_n[s:e][None, :] - chunk  # (K, B)
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
    log_num_k = chunked_log_numerator_k(u_kn, N_k, log_denom_n, chunk_size)
    return -1.0 * N_k * (1.0 - np.exp(f_k + log_num_k))


def chunked_mbar_objective_and_gradient(u_kn, N_k, f_k, chunk_size):
    """Equivalent to jax_mbar_objective_and_gradient (single log_denom pass)."""
    log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    obj = float(np.sum(log_denom_n) - np.dot(N_k, f_k))
    log_num_k = chunked_log_numerator_k(u_kn, N_k, log_denom_n, chunk_size)
    grad = -1.0 * N_k * (1.0 - np.exp(f_k + log_num_k))
    return obj, grad


def chunked_mbar_hessian(u_kn, N_k, f_k, chunk_size):
    """Equivalent to jax_mbar_hessian. Streams W_chunk = exp(...) of shape
    (B, K) and accumulates H += W_chunk.T @ W_chunk into a (K, K) buffer."""
    log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    K, _ = _shape_of(u_kn, N_k)
    dtype = _dtype_of(u_kn)
    H = np.zeros((K, K), dtype=dtype)
    W_sum = np.zeros(K, dtype=dtype)
    for s, chunk in _iter_chunks(u_kn, chunk_size):
        e = s + chunk.shape[1]
        W = np.exp(f_k[None, :] - chunk.T - log_denom_n[s:e, None])
        H += W.T @ W
        W_sum += W.sum(axis=0)
    H *= N_k
    H *= N_k[:, None]
    H -= np.diag(W_sum * N_k)
    return -H


def chunked_self_consistent_update(u_kn, N_k, f_k, chunk_size):
    """Equivalent to _jit_self_consistent_update."""
    log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    return -chunked_log_numerator_k(u_kn, N_k, log_denom_n, chunk_size)


def chunked_precondition_u_kn(u_kn, N_k, f_k, chunk_size):
    """Sample-wise shift of u_kn: ``u_kn -= u_kn.min(0); u_kn += log_denom - shift``.

    Dense path: mutates ``u_kn`` in-place, returns the same array.
    Callable path: returns a new callable that yields preconditioned chunks.

    MBAR is invariant under per-sample shifts so f_k and Log_W_nk are
    unaffected by either form.
    """
    if isinstance(u_kn, np.ndarray):
        u_min = u_kn.min(axis=0)  # (N,)
        np.subtract(u_kn, u_min, out=u_kn)
        log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
        shift = log_denom_n - float(np.dot(N_k, f_k) / N_k.sum())
        np.add(u_kn, shift, out=u_kn)
        return u_kn
    # Callable path: two streaming passes for u_min and log_denom, then
    # return a wrapping callable that applies the per-sample shift on read.
    _, N = _shape_of(u_kn, N_k)
    u_min = np.empty(N, dtype=_dtype_of(u_kn))
    for s, chunk in _iter_chunks(u_kn, chunk_size):
        u_min[s:s + chunk.shape[1]] = chunk.min(axis=0)
    log_denom_n = chunked_log_denominator(_shifted_provider(u_kn, -u_min),
                                           N_k, f_k, chunk_size)
    shift_n = (-u_min) + (log_denom_n - float(np.dot(N_k, f_k) / N_k.sum()))
    return _shifted_provider(u_kn, shift_n)


def _shifted_provider(base, shift):
    """Wrap a callable u_kn so each yielded chunk is offset by shift[s:s+B]."""
    def provider():
        s = 0
        for chunk in base():
            B = chunk.shape[1]
            yield chunk + shift[s:s + B]
            s += B
    return provider


def chunked_mbar_log_W_nk(u_kn, N_k, f_k, chunk_size):
    """Equivalent to jax_mbar_log_W_nk. The output is shape (N, K) so this
    only chunks the *peak* during construction, not the result.
    """
    log_denom_n = chunked_log_denominator(u_kn, N_k, f_k, chunk_size)
    K, N = _shape_of(u_kn, N_k)
    out = np.empty((N, K), dtype=_dtype_of(u_kn))
    for s, chunk in _iter_chunks(u_kn, chunk_size):
        e = s + chunk.shape[1]
        out[s:e] = f_k[None, :] - chunk.T - log_denom_n[s:e, None]
    return out
