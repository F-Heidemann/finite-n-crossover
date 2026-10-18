# Finite-sample Bayesian learning near singularities

**Francis J. Headley**

When does a Bayesian model begin to resolve a weak feature, and what does a learning-coefficient measurement tell us about that change? I study this in reduced-rank regression, where one layer can be integrated out exactly.

**[Notes (PDF)](notes.pdf)** · **[Proofs](proofs/proof_B9.pdf)**

## Main result

For a weak rank-one target of amplitude $\delta$, the relevant signal scale is $s=n\delta^2/2$. The proofs give an exact Gaussian reduction of the evidence, a uniform bound near zero truth when the dominant stratum is unique, and a logarithmic correction at a tie between strata:

$$
\frac{Z_n(C)}{Z_n(0)}
=
\Phi(s)\frac{\log[n/(1+s)]+O(1)}{\log n+O(1)}.
$$

For the amplitude-window marker in the $(N,H,M)=(2,1,2)$ model, the posterior-mass crossings are $s=1.75763,\ 1.64693,\ 1.57366$ at $n=10^4,10^8,10^{32}$, stable under grid refinement. At inverse temperature $\beta$, $n$ is replaced by $n\beta$, so what a learning-coefficient measurement resolves depends on the sampling temperature.

## Reproduce

```sh
python -m pip install -r requirements.txt
python review_checks/check_calculations.py
```

The script checks the exact evaluator against an independent integral (agreement about $4\times10^{-10}$), reproduces the window crossings under grid refinement, and runs a regression test for the sampled-data generator. It finishes in under a minute.

## Contents

| File | Contents |
|---|---|
| `notes.tex` / `.pdf` | The calculations, results, figures and open questions |
| `figures/` | Figures used in the notes |
| `proofs/proof_B9.tex` / `.pdf` | Uniform bound (unique stratum) and tie correction |
| `code/rrr.py` | Exact radial evaluator for the reduced-rank evidence |
| `code/a3_rrr_order.py` | Posterior-mass window calculations |
| `code/t1_mais_o78.py` | Sampled-data generator |
| `review_checks/check_calculations.py` | Deterministic checks of the results above |

## Prior work

Near-singularity scaling has prior art in [Watanabe and Amari (2002)](https://proceedings.neurips.cc/paper_files/paper/2002/file/c2ba1bc54b239208cb37b901c0d3b363-Paper.pdf); [Chen and Murfet (2025)](https://arxiv.org/abs/2504.18048) discuss the temperature dependence of learning-coefficient estimates. The contribution here is an explicit finite-sample calculation with uniform control near a singular truth.
