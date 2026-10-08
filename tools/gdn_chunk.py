# Check the chunked gated delta rule against the token-by-token recurrence.
import numpy as np
rng = np.random.default_rng(0)
d, T, C = 128, 200, 32
q = rng.standard_normal((T, d)); k = rng.standard_normal((T, d)); v = rng.standard_normal((T, d))
q /= np.linalg.norm(q, axis=1, keepdims=True); k /= np.linalg.norm(k, axis=1, keepdims=True)
q *= 1 / np.sqrt(d)
beta = 1 / (1 + np.exp(-rng.standard_normal(T)))
logg = -np.abs(rng.standard_normal(T)) * 0.5          # g = exp(logg) in (0, 1]
S0 = rng.standard_normal((d, d)) * 0.1
# sequential: S[i][j], i key, j value
S = S0.copy(); Oref = np.zeros((T, d))
for t in range(T):
    S = np.exp(logg[t]) * S
    delta = (v[t] - S.T @ k[t]) * beta[t]
    S = S + np.outer(k[t], delta)
    Oref[t] = S.T @ q[t]
Sref = S
# chunked
S = S0.copy(); O = np.zeros((T, d))
for c0 in range(0, T, C):
    c1 = min(c0 + C, T); n = c1 - c0
    K, Q, V, b = k[c0:c1], q[c0:c1], v[c0:c1], beta[c0:c1]
    G = np.cumsum(logg[c0:c1]); gam = np.exp(G)
    D = np.exp(G[:, None] - G[None, :])                # gamma_t / gamma_s
    L = np.tril(b[:, None] * D * (K @ K.T), -1)
    Tm = np.linalg.inv(np.eye(n) + L)
    U = Tm @ (b[:, None] * V); W = Tm @ ((b * gam)[:, None] * K)
    Dl = U - W @ S
    M = np.tril(D * (Q @ K.T))
    O[c0:c1] = gam[:, None] * (Q @ S) + M @ Dl
    S = gam[-1] * S + K.T @ ((np.exp(G[-1] - G))[:, None] * Dl)
print("max |O - Oref| / max |Oref| = %.2e" % (np.abs(O - Oref).max() / np.abs(Oref).max()))
print("max |S - Sref| / max |Sref| = %.2e" % (np.abs(S - Sref).max() / np.abs(Sref).max()))
