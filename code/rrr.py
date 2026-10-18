"""Numerical evaluator of the exact reduction Eq. (reduction) of outline Sec. 3.1.

Z_n(C) = c int_{R_+^q} prod rho_i^{|N-H|} prod_{i<l}|rho_i^2-rho_l^2| e^{-|rho|^2/2}
         prod (1+n rho_i^2)^{-M/2} e^{-tr S} 0F0^{(N)}(S, D_n(rho)) d rho,
D_n = diag(t_i), t_i = n rho_i^2/(1+n rho_i^2), S = (n/2) C C^T, q = min(N, H).

Angular factor Psi = e^{-tr S} 0F0^{(N)}(S, D):
  * rank-one S (eigenvalue s), any N:  0F0^{(N)}(S, D) = E_{w in S^{N-1}} exp(s sum_i t_i w_i^2)
      = sum_m prod_i (1/2)_{m_i} x_i^{m_i}/m_i! / (N/2)_{|m|},  x_i = s t_i  (confluent Lauricella)
      = Gamma(N/2) (2 pi i)^{-1} int_Br e^z z^{-N/2} prod_i (1 - x_i/z)^{-1/2} dz,
    evaluated with the fixed-Talbot rule (Abate-Valko 2004), checked against the positive
    series (q = 2) and against 0F0(S, P_j) = 1F1(j/2; N/2; s) (see selftest()).
  * N = 2, any S: 0F0^{(2)}(S, D) = exp(tr S tr D/2) I_0((s_1-s_2)(t_1-t_2)/2)   (exact).
lambda_n = -d log Z/d log n at fixed C is computed as minus the integrand-weighted average of the
pointwise derivative d log f/d log n (analytic), so no finite differences in n are needed.
"""
import numpy as np
from scipy.special import roots_legendre, gammaln, ive, hyp1f1
import mpmath as mp

# ---------------------------------------------------------------- strata bookkeeping
def lam_j(N, H, M, j):
    return (j * M + (H - j) * (N - j)) / 2


def strata(N, H, M):
    q = min(N, H)
    lj = [lam_j(N, H, M, j) for j in range(q + 1)]
    lmin = min(lj)
    js = [j for j in range(q + 1) if lj[j] == lmin]
    return lmin, js  # multiplicity m = len(js)


def Phi_rank1(s, N, H, M, j):
    """Stratum profile for rank-one C: E_{U,V} exp(-s tau_U tau_V)
    = 2F2((N-j)/2, (M-H+j)/2; N/2, M/2; -s)  (results_phase1_2.md, Eq. (Phi-rank1))."""
    a1, a2 = (N - j) / 2, (M - H + j) / 2
    if a1 == 0 or a2 <= 0:
        return mp.mpf(1)
    return mp.hyp2f2(a1, a2, mp.mpf(N) / 2, mp.mpf(M) / 2, -s)


def profile_rank1(s, N, H, M, j):
    """-s d/ds log Phi_j(s): predicted lambda_n(C) - lambda_n(0) in the double-scaling limit."""
    return float(-s * mp.diff(lambda x: mp.log(Phi_rank1(x, N, H, M, j)), s))


def profile_H1(s, N, j):
    """Outline (H1) prediction: -s d/ds log[e^{-s} 1F1(j/2; N/2; s)]."""
    if j == 0:
        return s
    return float(-s * mp.diff(lambda x: -x + mp.log(mp.hyp1f1(mp.mpf(j) / 2, mp.mpf(N) / 2, x)), s))


# ---------------------------------------------------------------- angular factor, rank-one S
_TM = 32
_th = np.arange(1, _TM) * np.pi / _TM
_cot = 1 / np.tan(_th)
_r = 2 * _TM / 5
_SK = _r * _th * (_cot + 1j)                       # Talbot nodes (t = 1)
_SIG = _th + (_th * _cot - 1) * _cot


def lauricella_talbot(x, N):
    """For x of shape (P, q), x >= 0: returns (log F, x_i d_i log F) with
    F = sum_m prod (1/2)_{m_i} x_i^{m_i}/m_i! / (N/2)_{|m|}."""
    x = np.asarray(x, float)
    b = N / 2
    xm = x.max(axis=1, keepdims=True)
    def G(w):  # w: (K,) complex -> (P, K)
        z = xm + w[None, :]
        lg = -b * np.log(z)
        for i in range(x.shape[1]):
            lg = lg - 0.5 * np.log1p(-x[:, i:i + 1] / z)
        return np.exp(lg), z
    g0, z0 = G(np.array([_r + 0j]))
    gk, zk = G(_SK)
    wk = np.exp(_SK) * (1 + 1j * _SIG)
    w0 = 0.5 * np.exp(_r)
    f = (_r / _TM) * (w0 * g0[:, 0].real + (gk * wk[None, :]).real.sum(1))
    dl = []
    for i in range(x.shape[1]):
        h0 = 0.5 * x[:, i:i + 1] / (z0 - x[:, i:i + 1])
        hk = 0.5 * x[:, i:i + 1] / (zk - x[:, i:i + 1])
        fi = (_r / _TM) * (w0 * (g0 * h0)[:, 0].real + (gk * hk * wk[None, :]).real.sum(1))
        dl.append(fi / f)
    logF = np.log(f) + xm[:, 0] + gammaln(b)
    return logF, np.stack(dl, 1)


def lauricella_series(x, N, K=None):
    """Positive series (exact up to truncation) for q <= 2; used to validate the Talbot rule."""
    x = np.atleast_2d(np.asarray(x, float))
    P, q = x.shape
    if K is None:
        K = int(x.max() + 12 * np.sqrt(x.max() + 1) + 40)
    m = np.arange(K)
    # log of (1/2)_m / m! = log Gamma(m+1/2) - log Gamma(1/2) - log m!
    la = gammaln(m + 0.5) - gammaln(0.5) - gammaln(m + 1)
    lpoch = gammaln(N / 2 + np.arange(2 * K)) - gammaln(N / 2)
    out = []
    for p in range(P):
        if q == 1:
            terms = la + m * np.log(x[p, 0] + 1e-300) - lpoch[m]
        else:
            t1 = la + m * np.log(x[p, 0] + 1e-300)
            t2 = la + m * np.log(x[p, 1] + 1e-300)
            terms = t1[:, None] + t2[None, :] - lpoch[m[:, None] + m[None, :]]
        mx = terms.max()
        out.append(mx + np.log(np.exp(terms - mx).sum()))
    return np.array(out)


# ---------------------------------------------------------------- radial quadrature
_XG, _WG = roots_legendre(16)


def panels(a, b, width):
    m = max(1, int(np.ceil((b - a) / width)))
    e = np.linspace(a, b, m + 1)
    lo, hi = e[:-1, None], e[1:, None]
    return (0.5 * (hi - lo) * _XG + 0.5 * (hi + lo)).ravel(), (0.5 * (hi - lo) * _WG + 0 * lo).ravel()


def radial_blocks(q, L, width=1.0, lo_pad=30.0, hi=4.0):
    """Yields blocks of nodes in log rho for the ordered region rho_1 > ... > rho_q, as
    (log rho (P,q), log weight (P,)). Coordinates u_1 = log rho_1, v_i = u_{i-1} - u_i > 0.
    The weight includes d rho = rho du and the factor q! for the ordering. One block per u_1 node."""
    u, wu = panels(-0.5 * L - lo_pad, hi, width)
    if q == 1:
        yield u[:, None], np.log(wu)
        return
    v, wv = panels(0.0, 0.5 * L + lo_pad + hi, width)
    vg = np.meshgrid(*([v] * (q - 1)), indexing="ij")
    wg = np.meshgrid(*([wv] * (q - 1)), indexing="ij")
    vcum = np.cumsum(np.stack([g.ravel() for g in vg], 1), axis=1)
    lwv = sum(np.log(g.ravel()) for g in wg) + np.log(float(np.prod(range(1, q + 1))))
    for ui, wi in zip(u, wu):
        U = np.concatenate([np.full((len(vcum), 1), ui), ui - vcum], 1)
        keep = U[:, -1] > -0.5 * L - lo_pad - 5
        yield U[keep], (lwv + np.log(wi))[keep]


def log_radial(U, n, N, H, M):
    """log of prod rho^{|N-H|} Vand e^{-|rho|^2/2} prod(1+n rho^2)^{-M/2} * prod rho (from d rho = rho du),
    and its log-n derivative. Also returns t_i."""
    rho2 = np.exp(2 * U)
    q = U.shape[1]
    lf = (abs(N - H) + 1) * U.sum(1) - 0.5 * rho2.sum(1) - (M / 2) * np.log1p(n * rho2).sum(1)
    for i in range(q):
        for l in range(i + 1, q):
            lf = lf + np.log(np.abs(rho2[:, i] - rho2[:, l]))
    t = n * rho2 / (1 + n * rho2)
    dlf = -(M / 2) * t.sum(1)
    return lf, dlf, t


def log_psi_rank1(s, t, N):
    """log Psi and d log Psi / d log n for rank-one S with eigenvalue s (s propto n)."""
    if s == 0:
        return np.zeros(len(t)), np.zeros(len(t))
    x = s * t
    if t.shape[1] == 1:
        xx = x[:, 0]
        lF = xx + np.log(hyp1f1((N - 1) / 2, N / 2, -xx))
        dl = xx * (1.0 / N) * hyp1f1((N - 1) / 2, N / 2 + 1, -xx) / hyp1f1((N - 1) / 2, N / 2, -xx)
        dl = dl[:, None]
    else:
        lF, dl = lauricella_talbot(x, N)
    dpsi = -s + (dl * (2 - t)).sum(1)  # d x_i/dL = x_i (2 - t_i)
    return -s + lF, dpsi


def log_psi_N2(s1, s2, t):
    """N = 2, any S = diag(s1, s2) (both propto n); t has 1 or 2 columns."""
    t1 = t[:, 0]
    t2 = t[:, 1] if t.shape[1] > 1 else np.zeros_like(t1)
    a, b = s1 + s2, s1 - s2
    T, Dl = t1 + t2, t1 - t2
    z = b * Dl / 2
    lpsi = -a + a * T / 2 + np.log(ive(0, z)) + np.abs(z)
    dT = t1 * (1 - t1) + t2 * (1 - t2)
    dD = t1 * (1 - t1) - t2 * (1 - t2)
    ratio = ive(1, z) / ive(0, z)
    dpsi = -a + a * T / 2 + a * dT / 2 + ratio * (z + b * dD / 2)
    return lpsi, dpsi


def lam_n(n, N, H, M, S, width=1.0, return_logZ=False, prune=40.0, lo_pad=30.0):
    """lambda_n(C). S = ('rank1', s) or ('N2', s1, s2). Also returns log Z_n/c if asked.
    Nodes whose radial weight is below e^{-prune - tr S} of the largest are dropped
    (the angular factor Psi lies in [e^{-tr S}, 1], so the dropped mass is < e^{-prune} per node)."""
    q = min(N, H)
    L = np.log(n)
    trS = S[1] if S[0] == "rank1" else S[1] + S[2]
    gmax = max(np.max(log_radial(U, n, N, H, M)[0] + lw) for U, lw in radial_blocks(q, L, width, lo_pad))
    num = den = 0.0
    shift = None
    for U, lw in radial_blocks(q, L, width, lo_pad):
        lf, dlf, t = log_radial(U, n, N, H, M)
        keep = lf + lw > gmax - trS - prune
        if not keep.any():
            continue
        lf, dlf, t, lwc = lf[keep], dlf[keep], t[keep], lw[keep]
        if S[0] == "rank1":
            lp, dp = log_psi_rank1(S[1], t, N)
        else:
            lp, dp = log_psi_N2(S[1], S[2], t)
        lg = lf + lp + lwc
        if shift is None:
            shift = gmax
        p = np.exp(lg - shift)
        num += np.sum(p * (dlf + dp))
        den += np.sum(p)
    lam = -num / den
    if return_logZ:
        return lam, np.log(den) + shift
    return lam


def selftest():
    rng = np.random.default_rng(0)
    worst = 0.0
    for N in (2, 3, 5):
        x = rng.uniform(0, 60, size=(200, 2))
        x[:20, 1] = 0
        a = lauricella_talbot(x, N)[0]
        b = lauricella_series(x, N)
        worst = max(worst, np.max(np.abs(a - b)))
    print(f"Talbot vs positive series, q=2, N=2,3,5, x in [0,60]: max |dlog F| = {worst:.2e}")
    worst = 0.0
    for N in (3, 4, 5):
        for j in (1, 2, 3):
            if j > N:
                continue
            for s in (0.3, 3.0, 30.0, 80.0):
                a = lauricella_talbot(np.full((1, j), s), N)[0][0]
                b = float(mp.log(mp.hyp1f1(mp.mpf(j) / 2, mp.mpf(N) / 2, s)))
                worst = max(worst, abs(a - b))
    print(f"Talbot vs 1F1(j/2;N/2;s) (projector identity), j<=3, s<=80: max |dlog F| = {worst:.2e}")
    # derivative check
    x = rng.uniform(0, 40, size=(50, 3))
    lF, dl = lauricella_talbot(x, 5)
    h = 1e-5
    err = 0.0
    for i in range(3):
        xp, xm = x.copy(), x.copy()
        xp[:, i] *= np.exp(h)
        xm[:, i] *= np.exp(-h)
        fd = (lauricella_talbot(xp, 5)[0] - lauricella_talbot(xm, 5)[0]) / (2 * h)
        err = max(err, np.max(np.abs(fd - dl[:, i])))
    print(f"Talbot x_i d_i log F vs finite difference, q=3: max err = {err:.2e}")


if __name__ == "__main__":
    selftest()
