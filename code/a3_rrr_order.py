"""A3 tasks 5-6 (RRR part): posterior-mass order parameter and local free energies, RRR with H = 1, N = 2.

Model y = a b^T x + eps, a in R^2, b in R^M, standard Gaussian priors; orthogonal design, so the
posterior is  p(a, b | data) ∝ exp(-(n/2)|a b^T - C_hat|_F^2) phi(a) phi(b),  C_hat = C + E/sqrt(n)
(outline Sec. 4.2; a2_identity.py). Truth C = sigma e f^T, s = n sigma^2/2.
Fitted amplitude along the truth: alpha = e^T a b^T f = (e.a)(f.b); the truth has alpha = sigma.
Given a, b is Gaussian (precision (1 + n|a|^2) I, mean n C_hat^T a/(1 + n|a|^2)), so alpha | a is Gaussian;
the a-marginal is a 2-D quadrature (polar, u = log|a|).
Recorded per (n, s, draw): log Z_n(C_hat) (Z normalised as E_prior), P(|alpha| < sigma/2),
P(|alpha - sigma| < sigma/2), P(alpha > sigma/2). Noise-free: E = 0.

Usage: python3 a3_rrr_order.py M K   (K noise draws, even; antithetic)
"""
import sys
import numpy as np
from multiprocessing import Pool
from scipy.special import ndtr, logsumexp

NU, NTH = 1100, 192
SS = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.3, 16.0, 22.6, 32.0, 64.0, 128.0, 256.0])
NS = [1e2, 1e4, 1e8]


def post(n, M, Chat, sigma):
    L = np.log(n)
    u = np.linspace(-0.5 * L - 14, 3.0, NU); du = u[1] - u[0]
    th = np.arange(NTH) * 2 * np.pi / NTH
    rho = np.exp(u)[:, None]
    a = np.stack([rho * np.cos(th)[None], rho * np.sin(th)[None]], -1)            # (NU, NTH, 2)
    q = 1 + n * rho ** 2
    v = a @ Chat                                                                   # C_hat^T a, (NU, NTH, M)
    lw = (2 * u[:, None] - rho ** 2 / 2 - np.log(2 * np.pi) - (M / 2) * np.log(q)
          + n ** 2 * np.sum(v * v, -1) / (2 * q) - n * np.sum(Chat ** 2) / 2)
    lZ = logsumexp(lw) + np.log(du * 2 * np.pi / NTH)
    p = np.exp(lw - logsumexp(lw))
    ea = a[..., 0]                                                                 # e = e1
    mu = ea * n * v[..., 0] / q                                                    # f = e1: (f.b) mean
    sd = np.maximum(np.abs(ea) / np.sqrt(q), 1e-300)
    def P(lo, hi):
        return np.sum(p * (ndtr((hi - mu) / sd) - ndtr((lo - mu) / sd)))
    if sigma == 0:
        return lZ, 0.0, 0.0, P(0, np.inf)
    return lZ, P(-sigma / 2, sigma / 2), P(sigma / 2, 1.5 * sigma), P(sigma / 2, np.inf)


def work(args):
    n, s, M, Es = args
    sigma = np.sqrt(2 * s / n)
    C = np.zeros((2, M)); C[0, 0] = sigma
    return np.array([post(n, M, C + E / np.sqrt(n), sigma) for E in Es])


if __name__ == "__main__":
    M, K = int(sys.argv[1]), int(sys.argv[2])
    rng = np.random.default_rng(777)
    E = rng.standard_normal((K // 2, 2, M)); E = np.concatenate([E, -E])
    E = np.concatenate([np.zeros((1, 2, M)), E])                                   # draw 0: noise-free
    chunks = np.array_split(E, 10)
    jobs = [(n, s, M, ch) for n in NS for s in SS for ch in chunks]
    with Pool(10) as p:
        res = p.map(work, jobs)
    R = np.array([np.concatenate(res[i * 10:(i + 1) * 10]) for i in range(len(NS) * len(SS))]).reshape(len(NS), len(SS), K + 1, 4)
    np.savez(f"out/a3_rrr_order_M{M}.npz", R=R, NS=NS, SS=SS)
    print("saved", R.shape)
