"""Chunked working-array variants of the MBAR inner-fit operations.

The default `mbar_solvers.py` calls

    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)

which materializes a (N, K) working array the same size as ``u_kn``. For
large N, this is the dominant peak-memory cost during the fit -- not
``u_kn`` itself. See pymbar issue #574.

These helpers compute the same quantities by streaming over the sample
axis, with peak working-array memory ``O(chunk_size * K)`` instead of
``O(N * K)``.

The helpers accept either a dense ``(K, N)`` ndarray or a zero-arg
callable yielding ``(K, B)`` chunks (so even ``u_kn`` itself can be
streamed from disk).

When JAX is importable (and ``PYMBAR_DISABLE_JAX`` is unset), the
per-chunk inner kernels run jit-compiled at fixed ``(K, chunk_size)``
shape; otherwise the same code runs as plain numpy. Last chunks shorter
than ``chunk_size`` are padded with ``+inf`` so the same compiled kernel
applies, then the valid prefix is sliced on output.
"""
import os

import numpy as np

try:
    if os.environ.get("PYMBAR_DISABLE_JAX", "").lower() in ("true", "yes", "1"):
        raise ImportError("JAX disabled by PYMBAR_DISABLE_JAX")
    from jax import config as _jax_config
    if not _jax_config.x64_enabled:
        _jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import jit
    _USE_JIT = True
except ImportError:
    _USE_JIT = False
    jnp = np

    def jit(fn):
        return fn


# =============================================================================
# Per-chunk inner kernels. JIT-compiled at fixed (K, chunk_size) shape when JAX
# is available; same code paths execute as plain numpy when not. Outer loops
# call these once per chunk, padding the last chunk with +inf so the cached
# compile applies (the +inf entries contribute 0 to logsumexp / W).
# =============================================================================


@jit
def _kernel_log_denom(a, chunk):
    """logsumexp(a - chunk.T, axis=1). a: (K,), chunk: (K, B). returns (B,)."""
    z = a[:, None] - chunk
    m = z.max(axis=0)
    return m + jnp.log(jnp.exp(z - m).sum(axis=0))


@jit
def _kernel_log_num_update(m_k, S_k, log_denom_chunk, chunk):
    """One streaming step of running max-shift accumulator for log_numerator_k.

    Updates (m_k, S_k) given a new chunk. Returns (m_new, S_new).
    """
    z = -log_denom_chunk[None, :] - chunk  # (K, B)
    m_chunk = z.max(axis=1)
    m_new = jnp.maximum(m_k, m_chunk)
    finite = jnp.isfinite(m_new)
    scale = jnp.where(finite, jnp.exp(m_k - m_new), 0.0)
    contrib = jnp.where(finite[:, None], jnp.exp(z - m_new[:, None]), 0.0).sum(axis=1)
    return m_new, S_k * scale + contrib


@jit
def _kernel_hessian_update(H, W_sum, f_k, chunk, log_denom_chunk):
    """One streaming step of Hessian Gram accumulation. Returns (H_new, W_sum_new)."""
    W = jnp.exp(f_k[None, :] - chunk.T - log_denom_chunk[:, None])  # (B, K)
    return H + W.T @ W, W_sum + W.sum(axis=0)


@jit
def _kernel_log_W_chunk(f_k, chunk, log_denom_chunk):
    """log_W chunk: f_k - chunk.T - log_denom_chunk[:, None]. Returns (B, K)."""
    return f_k[None, :] - chunk.T - log_denom_chunk[:, None]


def _pad_to(chunk, target_B):
    """Pad chunk (K, B) to (K, target_B) with +inf so logsumexp ignores padded
    samples (exp(-inf) = 0). Pass-through when shapes match."""
    B = chunk.shape[1]
    if B == target_B:
        return chunk
    return np.pad(chunk, ((0, 0), (0, target_B - B)), constant_values=np.inf)


def _pad_ld(ld, target_B, dtype):
    """Pad log_denominator_n slice (B,) to (target_B,) with zeros. Padded
    entries are arbitrary: the chunk's +inf padding forces z = -ld - chunk
    to -inf, so exp contribution is 0 regardless of ld value."""
    B = ld.shape[0]
    if B == target_B:
        return ld
    return np.concatenate([ld, np.zeros(target_B - B, dtype=dtype)])


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
        B = chunk.shape[1]
        res = _kernel_log_denom(a, _pad_to(chunk, chunk_size))
        out[s:s + B] = np.asarray(res[:B])
    return out


def _chunked_log_num_rows(u_arr, log_denominator_n, chunk_size, R):
    """Streaming logsumexp(-u_arr - log_denominator_n, axis=1) for an R-row
    matrix. Per-row running max-shift / sum across chunks. Shared by
    chunked_log_numerator_k (sampled states; R from N_k for callable u_kn)
    and chunked_log_numerator_l (perturbed states; R from u_ln.shape[0]).
    """
    dtype = _dtype_of(u_arr)
    m = np.full(R, -np.inf, dtype=dtype)
    S = np.zeros(R, dtype=dtype)
    for s, chunk in _iter_chunks(u_arr, chunk_size):
        B = chunk.shape[1]
        ld = _pad_ld(log_denominator_n[s:s + B], chunk_size, dtype)
        m, S = _kernel_log_num_update(m, S, ld, _pad_to(chunk, chunk_size))
    return np.array(m) + np.log(np.array(S))


def chunked_log_numerator_k(u_kn, N_k, log_denominator_n, chunk_size):
    """Streaming logsumexp(-log_denominator_n - u_kn, axis=1).

    log_num_k[k] = log sum_n exp(-log_denominator_n[n] - u_kn[k, n])
    """
    K, _ = _shape_of(u_kn, N_k)
    return _chunked_log_num_rows(u_kn, log_denominator_n, chunk_size, K)


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
        B = chunk.shape[1]
        ld = _pad_ld(log_denom_n[s:s + B], chunk_size, dtype)
        H, W_sum = _kernel_hessian_update(H, W_sum, f_k, _pad_to(chunk, chunk_size), ld)
    H = np.array(H, dtype=dtype)
    W_sum = np.array(W_sum, dtype=dtype)
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
    dtype = _dtype_of(u_kn)
    out = np.empty((N, K), dtype=dtype)
    for s, chunk in _iter_chunks(u_kn, chunk_size):
        B = chunk.shape[1]
        ld = _pad_ld(log_denom_n[s:s + B], chunk_size, dtype)
        res = _kernel_log_W_chunk(f_k, _pad_to(chunk, chunk_size), ld)
        out[s:s + B] = np.asarray(res[:B])
    return out


# =============================================================================
# Batch-evaluation helpers (point-estimate path for compute_expectations and
# compute_perturbed_free_energies). Same _kernel_log_num_update as the fit-side
# helpers; row dimension is now perturbed states (no N_k weighting) or
# (state, observable) pairs from a state_map.
# =============================================================================


def chunked_log_numerator_l(u_ln, log_denominator_n, chunk_size):
    """Streaming logsumexp(-u_ln - log_denominator_n, axis=1) for perturbed
    (unsampled) states. Used by compute_expectations_inner for log_C_a[l].
    """
    return _chunked_log_num_rows(u_ln, log_denominator_n, chunk_size,
                                 u_ln.shape[0])


def chunked_log_observable(u_ln, log_A_n, state_map, log_denominator_n,
                           chunk_size):
    """Streaming logsumexp(log_A_n[i, n] - u_ln[l, n] - log_denominator_n[n], axis=n)
    for each (l, i) = (state_map[0, s], state_map[1, s]) over s = 0..S-1.

    log_obs_s[s] = log sum_n exp(log_A_n[state_map[1, s], n]
                                 - u_ln[state_map[0, s], n]
                                 - log_denominator_n[n])

    Used by compute_expectations_inner: f_k[K + NL + s] = -log_obs_s[s].
    Peak working memory O(S * chunk_size); pick chunk_size so this stays
    below the dense-path footprint S * N.
    """
    S = state_map.shape[1]
    dtype = _dtype_of(u_ln)
    l_idx = state_map[0, :]
    i_idx = state_map[1, :]
    m_s = np.full(S, -np.inf, dtype=dtype)
    Sum_s = np.zeros(S, dtype=dtype)
    for s, u_chunk in _iter_chunks(u_ln, chunk_size):
        B = u_chunk.shape[1]
        # _kernel_log_num_update computes z = -log_denom - chunk; we want
        # z[t, n] = log_A_n[i_t, n] - u_ln[l_t, n] - log_denom[n], so feed
        # chunk_eff = u_ln[l_t] - log_A_n[i_t].
        chunk_eff = (u_chunk[l_idx] - log_A_n[i_idx, s:s + B]).astype(dtype)
        ld = _pad_ld(log_denominator_n[s:s + B], chunk_size, dtype)
        m_s, Sum_s = _kernel_log_num_update(
            m_s, Sum_s, ld, _pad_to(chunk_eff, chunk_size))
    return np.array(m_s) + np.log(np.array(Sum_s))
