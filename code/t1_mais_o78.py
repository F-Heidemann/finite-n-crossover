"""T1: MAIS-O78(a) in its own setting. N = M = H = 2, target Phi = delta e1 e1^T (rank r = 1 < H), y = B A x + eps,
A in R^{H x M}, B in R^{N x H} (MAIS convention), x ~ N(0, I).

Claim tested. O78(a) asserts that, outside windows of width c sqrt(n log n) around the crossover
n_1 / log n_1 = 2 dlam / delta^2 (i.e. s = n delta^2/2 = dlam log n, dlam = lambda_1 - lambda_0 = 1/2 here),
the posterior puts maximal window mass on the saddle set C_{k*}, k* = argmin_k [n L_k + lambda_k log n].
Below the crossover k* = 0. We compute posterior masses of the function-space windows
    W_k(rho) = { ||B A - Phi_k||_F < rho },  Phi_0 = 0, Phi_1 = Phi,  rho = delta/2 and delta/4,
and the s at which mass(W_1) overtakes mass(W_0).

Posterior. Data enter through (Phi_hat, Sigma_hat) (outline Sec. 4.2):
    log L(A, B) = -(n/2) tr[(BA - Phi_hat) Sigma_hat (BA - Phi_hat)^T] + const.
Sigma_hat ~ Wishart(n, I)/n (Bartlett), rows of Phi_hat - Phi ~ N(0, Sigma_hat^{-1}/n). Noise-free: Phi_hat = Phi, Sigma_hat = I.
Rows b_i of B are independent given A: b_i | A ~ N(P^{-1} h_i, P^{-1}), P = I + n A Sigma_hat A^T, h_i = n A Sigma_hat phi_i^T.
B is integrated exactly; A (4-d) by importance sampling in SVD coordinates A = R(t1) diag(e^u1, +-e^u2) R(t2)^T,
proposal uniform in (u1 > u2, sign, t1, t2), u in [-L/2 - 7, log 4]; Jacobian |rho1^2 - rho2^2| rho1 rho2.
Window masses use KB exact draws of B | A per A sample.
Priors: 'gauss' (standard normal entries) and 'box' (standard normal truncated to [-R, R]^8, R = 1.5: compact,
smooth and positive on W; meets every C_k and the origin). For 'box', A outside the box gets weight 0 and the
B-truncation enters through the indicator of each B draw (unbiased pseudo-marginal estimate).

Validation (part v): noise-free Gaussian log Z_n(s) - log Z_n(0) against the exact evaluator rrr.lam_n.
Usage: python3 t1_mais_o78.py v | nf | radius | noisy
"""
import sys
import numpy as np

N = M = H = 2
DLAM = 0.5
R_BOX = 1.5


class Sampler:
    def __init__(self, n, NA, seed):
        rng = np.random.default_rng(seed)
        L = np.log(n)
        lo, hi = -0.5 * L - 7.0, np.log(4.0)
        u = rng.uniform(lo, hi, (NA, 2)); u.sort(1)
        u1, u2 = u[:, 1], u[:, 0]
        sg = rng.choice([-1.0, 1.0], NA)
        t1, t2 = rng.uniform(0, 2 * np.pi, (2, NA))
        r1, r2 = np.exp(u1), sg * np.exp(u2)
        c1, s1, c2, s2 = np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2)
        U = np.stack([np.stack([c1, -s1], -1), np.stack([s1, c1], -1)], -2)
        V = np.stack([np.stack([c2, -s2], -1), np.stack([s2, c2], -1)], -2)
        D = np.zeros((NA, 2, 2)); D[:, 0, 0] = r1; D[:, 1, 1] = r2
        self.A = U @ D @ np.swapaxes(V, 1, 2)
        self.ljac = np.log(np.abs(r1 ** 2 - r2 ** 2)) + u1 + u2
        self.n, self.rng = n, rng

    def run(self, Phi_hat, Sig, prior, delta, KB=4, rhos=(0.5, 0.25)):
        n, A = self.n, self.A
        lp = -0.5 * np.sum(A * A, (1, 2)) + self.ljac
        if prior == "box":
            lp = np.where(np.all(np.abs(A) <= R_BOX, (1, 2)), lp, -np.inf)
        # stable form: At = A Sig^{1/2} = U diag(sv) W^T; P = U diag(1 + n sv^2) U^T
        es, Es = np.linalg.eigh(Sig); Sh = (Es * np.sqrt(es)) @ Es.T
        U, sv, Wt = np.linalg.svd(A @ Sh)
        ev = 1 + n * sv ** 2                                                  # (NA, H)
        psi = Sh @ Phi_hat.T                                                  # (M, N): column i = Sig^{1/2} phi_i
        c = Wt @ psi                                                          # (NA, H, N): W^T psi_i
        lw = lp - 0.5 * n * np.sum(c ** 2 / ev[:, :, None], (1, 2)) - (N / 2) * np.sum(np.log(ev), 1)
        mu = U @ ((n * sv / ev)[:, :, None] * c)                              # (NA, H, N): column i = mean of b_i
        Lc = U / np.sqrt(ev)[:, None, :]                                      # Lc Lc^T = P^{-1}
        good = np.isfinite(lw)
        mx = lw[good].max()
        w = np.where(good, np.exp(lw - mx), 0.0)
        logZ = np.log(w.mean()) + mx
        # draws of B | A
        Phi0 = np.zeros((N, M)); Phi1 = np.zeros((N, M)); Phi1[0, 0] = delta
        acc = {"box": 0.0}
        tot = np.zeros(1 + 2 * len(rhos)); wsum = 0.0
        for _ in range(KB):
            z = self.rng.standard_normal((len(A), H, N))
            Bt = mu + Lc @ z                                                  # (NA, H, N): B^T
            B = np.swapaxes(Bt, 1, 2)
            ind = np.ones(len(A)) if prior == "gauss" else np.all(np.abs(B) <= R_BOX, (1, 2)).astype(float)
            F = B @ A
            d0 = np.sqrt(np.sum((F - Phi0) ** 2, (1, 2))); d1 = np.sqrt(np.sum((F - Phi1) ** 2, (1, 2)))
            cols = [ind]
            for r in rhos:
                cols += [ind * (d0 < r * delta), ind * (d1 < r * delta)]
            tot += np.array([np.sum(w * c) for c in cols]); wsum += 1
        ess = w.sum() ** 2 / np.sum(w * w)
        # log Z for box prior: multiply by mean truncation indicator
        logZ_eff = logZ + np.log(max(tot[0] / (wsum * w.sum()), 1e-300))
        masses = tot[1:] / tot[0]
        return logZ_eff, masses, ess


def dataset(n, delta, rng):
    Phi = np.zeros((N, M)); Phi[0, 0] = delta
    # Bartlett: Sigma_hat = T T^T / n, T lower triangular
    T = np.zeros((M, M))
    for i in range(M):
        T[i, i] = np.sqrt(rng.chisquare(n - i))
        T[i, :i] = rng.standard_normal(i)
    Sig = T @ T.T / n
    Lc = np.linalg.cholesky(Sig)
    Z = rng.standard_normal((N, M))
    Phi_hat = Phi + np.linalg.solve(Lc.T, Z.T).T / np.sqrt(n)          # rows ~ N(0, Sig^{-1}/n)
    return Phi_hat, Sig


SS = np.geomspace(0.1, 40, 22)
SSN = np.geomspace(0.1, 12, 15)   # noisy runs: bracket the crossing only


def crossing(s, y):
    k = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    if len(k) == 0:
        return np.nan
    k = k[0]
    return np.exp(np.log(s[k]) + (0 - y[k]) * (np.log(s[k + 1]) - np.log(s[k])) / (y[k + 1] - y[k]))


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "v":
        import rrr
        for n in (1e4, 1e8):
            S = Sampler(n, 400_000, 1)
            l0 = S.run(np.zeros((N, M)), np.eye(M), "gauss", 1e-9)[0]
            e0 = rrr.lam_n(n, 2, 2, 2, ("rank1", 0.0), return_logZ=True)[1]
            for s in (0.5, 2.0, 8.0, 32.0):
                d = np.sqrt(2 * s / n); Ph = np.zeros((N, M)); Ph[0, 0] = d
                l, m, ess = S.run(Ph, np.eye(M), "gauss", d)
                e = rrr.lam_n(n, 2, 2, 2, ("rank1", s), return_logZ=True)[1]
                print(f"n={n:.0e} s={s:5.1f}: IS logZ(s)-logZ(0) = {l - l0:+.4f}   exact = {e - e0:+.4f}   ESS = {ess:,.0f}", flush=True)
    elif part == "nf":
        save = {}
        for prior in ("gauss", "box"):
            print(f"noise-free, prior={prior}: s at which mass(W_1) overtakes mass(W_0), vs O78 crossover s = dlam log n")
            for n in (1e4, 1e6, 1e8, 1e12, 1e16):
                S = Sampler(n, 300_000, 2)
                rows = []
                for s in SS:
                    d = np.sqrt(2 * s / n); Ph = np.zeros((N, M)); Ph[0, 0] = d
                    rows.append(S.run(Ph, np.eye(M), prior, d)[1:])
                m = np.array([r[0] for r in rows]); essv = np.array([r[1] for r in rows])
                save[f"{prior}_{int(round(np.log10(n)))}"] = m
                sx = [crossing(SS, m[:, 2 * k + 1] - m[:, 2 * k]) for k in range(2)]
                i3 = np.argmin(np.abs(SS - DLAM * np.log(n) / 2))
                print(f"  n=1e{int(round(np.log10(n))):<3d} crossing s: rho=delta/2 {sx[0]:6.3f}, rho=delta/4 {sx[1]:6.3f};  "
                      f"O78 s = {DLAM*np.log(n):5.2f};  at s={SS[i3]:.2f} (about half the O78 value): "
                      f"W0={m[i3,0]:.3f} W1={m[i3,1]:.3f};  ESS at crossing {essv[np.argmin(np.abs(SS-sx[0]))]:,.0f}, at s={SS[i3]:.2f} {essv[i3]:,.0f}", flush=True)
        np.savez("out/t1_noisefree.npz", SS=SS, **save)
    elif part == "radius":
        # crossing s against window radius rho = delta / 2^k: heuristic mass(W1)/mass(W0) ~ (rho/delta)^(2 dlam) e^s,
        # so s_cross ~ 2 dlam log(delta/rho) + O(log log), independent of n; rho ~ delta/sqrt(n) would give dlam log n.
        ks = np.arange(1, 6); rhos = tuple(0.5 ** ks)
        SR = np.geomspace(0.2, 30, 26)
        out = {}
        print("noise-free, Gaussian prior: crossing s of mass(W_1) = mass(W_0) against rho/delta")
        for n in (1e6, 1e12):
            S = Sampler(n, 300_000, 4)
            m = []
            for s in SR:
                d = np.sqrt(2 * s / n); Ph = np.zeros((N, M)); Ph[0, 0] = d
                m.append(S.run(Ph, np.eye(M), "gauss", d, KB=16, rhos=rhos)[1])
            m = np.array(m); out[f"n{int(round(np.log10(n)))}"] = m
            for j, k in enumerate(ks):
                sx = crossing(SR, m[:, 2 * j + 1] - m[:, 2 * j])
                i = np.argmin(np.abs(SR - sx)) if np.isfinite(sx) else 0
                print(f"  n=1e{int(round(np.log10(n))):<3d} rho=delta/{2**k:<3d} s_cross={sx:6.3f}  2*dlam*log(delta/rho)={2*DLAM*np.log(2**k):5.2f}  "
                      f"window masses at crossing ~{m[i, 2*j]:.1e}", flush=True)
        np.savez("out/t1_radius.npz", SR=SR, ks=ks, **out)
    elif part == "noisy":
        K = 32
        print(f"sampled data, {K} data sets per n; Gaussian and box priors; window radius delta/2")
        for prior in ("gauss", "box"):
            for n in (1e4, 1e6, 1e8, 1e12, 1e16):
                S = Sampler(n, 200_000, 3)
                s_half = DLAM * np.log(n) / 2
                cross, win_half = [], []
                for k in range(K):
                    m = []
                    for s in SSN:
                        d = np.sqrt(2 * s / n)
                        Ph, Sig = dataset(n, d, np.random.default_rng(1000 * k + 7))   # same noise across s
                        m.append(S.run(Ph, Sig, prior, d, KB=2, rhos=(0.5,))[1])
                    m = np.array(m)
                    cross.append(crossing(SSN, m[:, 1] - m[:, 0]))
                    d = np.sqrt(2 * s_half / n)
                    Ph, Sig = dataset(n, d, np.random.default_rng(1000 * k + 7))
                    mh = S.run(Ph, Sig, prior, d, KB=2, rhos=(0.5,))[1]
                    win_half.append(mh[1] > mh[0])
                cross = np.array(cross)
                print(f"  {prior:5s} n=1e{int(round(np.log10(n))):<3d} crossing s: median {np.nanmedian(cross):.2f}, "
                      f"IQR [{np.nanpercentile(cross,25):.2f}, {np.nanpercentile(cross,75):.2f}];  O78 s = {DLAM*np.log(n):5.2f};  "
                      f"P(W1 > W0) at s = {s_half:.2f} (half the O78 value): {np.mean(win_half):.2f} ({K} sets)", flush=True)
