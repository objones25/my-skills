# Random Numbers, Distributions, and Statistics

`numpy.random`, `scipy.stats`, and the two smaller modules worth knowing about,
`scipy.special` and `scipy.spatial`. Open this for reproducible random draws, distribution
objects, hypothesis tests, resampling, and the numerically-safe versions of common
expressions.

## Random numbers: `Generator`, not `RandomState`

```python
import numpy as np
rng = np.random.default_rng(42)
```

That is the whole modern API. `np.random.seed()` and the bare module-level functions
(`np.random.rand`, `np.random.randn`, `np.random.randint`) mutate a hidden global state,
which makes results depend on execution order and makes library code and user code fight
over the same stream. `default_rng(seed)` returns an isolated `Generator` you pass around
explicitly.

```
rng.integers(0, 10, 5)          [0 7 6 4 4]      half-open, replaces randint
rng.random(3)                   [0.697368 0.094177 0.975622]
rng.normal(0, 1, 3)             [ 0.12784  -0.316243 -0.016801]
rng.choice(10, 4, replace=False)  [4 5 6 7]
rng.spawn(2)                    two independent child Generators
```

Name changes from the legacy API: `randint` -> `integers` (and it is half-open, so
`integers(0, 10)` never returns 10, whereas the legacy `randint` had the same convention but
`random_integers` did not), `rand` -> `random`, `randn` -> `standard_normal`,
`random_sample` -> `random`.

`rng.spawn(k)` is how you get independent streams for parallel workers. Passing the same
`rng` to several processes gives every process the same numbers. Seeding worker `i` with
`default_rng(root_seed + i)` gives overlapping seed sets across runs with different root seeds
(numpy's parallel-RNG guide marks this "UNSAFE"); `default_rng([i, root_seed])` is the safe
manual form. `spawn` is the supported way.

The same `rng=` keyword runs through the newer `scipy` functions (`stats.bootstrap`,
`stats.permutation_test`, `optimize.differential_evolution`, `sparse.linalg.eigsh`,
`sparse.random_array`; added in 1.15, and in 1.17 for `eigsh`). The legacy distribution
methods are behind: `stats.norm.rvs(size=3, rng=rng)` raises `TypeError` on scipy 1.17 and
1.18, so those take `random_state=rng`.

For reproducibility in a script: create one `rng` at the top, thread it through, and record
the seed. Do not seed inside a function.

## Distributions

Every distribution in `scipy.stats` shares one interface. `loc` shifts, `scale` stretches,
and any shape parameters come first.

```python
from scipy import stats
d = stats.norm(loc=2, scale=3)     # "frozen": parameters fixed, methods take only x
```

```
mean 2.000  var 9.000  pdf(2) 0.132981  cdf(2) 0.500  ppf(0.975) 7.879892
```

| method | returns |
|---|---|
| `pdf(x)` / `pmf(k)` | density / probability mass |
| `logpdf(x)` | log density, computed stably |
| `cdf(x)` | `P(X <= x)` |
| `sf(x)` | `P(X > x)`, the survival function |
| `ppf(q)` | inverse CDF (quantile) |
| `isf(q)` | inverse survival function |
| `rvs(size=, random_state=)` | random draws |
| `fit(data)` | maximum-likelihood parameter estimates |
| `mean/var/std/median/interval/stats/entropy` | summaries |

**Use `sf`, not `1 - cdf`, in the upper tail.** `cdf` saturates at 1.0 and the subtraction
loses everything:

```
z= 5   1-cdf = 2.866516e-07    sf = 2.866516e-07
z=10   1-cdf = 0.000000e+00    sf = 7.619853e-24
```

Same for `isf(q)` versus `ppf(1-q)`. This is the cancellation problem from
`04-floating-point.md` in its most common disguise.

`fit` returns shape parameters, then `loc`, then `scale`. It fits `loc` by default even for
distributions defined on `[0, inf)`, which usually is not what you want:

```
500 draws from gamma(a=2, scale=1.5), random_state=np.random.default_rng(1)
gamma.fit(data)             [ 2.4312  -0.0609   1.1893]     loc drifted negative
gamma.fit(data, floc=0)     [ 2.2650   0.0000   1.2496]     loc pinned
```

Fix parameters you know with `f<name>=`: `floc=0`, `fscale=1`, `fa=2`.

`stats.fit(dist, data, bounds=...)` is the newer, more controllable fitting entry point, and
`stats.goodness_of_fit` tests the result.

`scipy.stats` also ships a newer object-oriented distribution layer: `stats.Normal`,
`stats.Uniform`, `stats.Mixture`, and `stats.make_distribution(dist)` to convert an old-style
distribution into one. These support arithmetic on distribution objects and have cleaner
parameterization (`stats.Normal(mu=2.0, sigma=3.0)`). The classic `stats.norm` interface is
not going anywhere; the new one is worth using for anything involving transformed or mixed
distributions.

Multivariate: `multivariate_normal`, `dirichlet`, `wishart`, `multinomial`. These take
`mean`/`cov` rather than `loc`/`scale` and are frozen the same way.

## Hypothesis tests

Every test returns a result object with at least `.statistic` and `.pvalue`, and most now
carry `.confidence_interval()`.

```
ttest_ind(a, b)   statistic -1.7319   p 0.086431   CI [-0.5637, 0.0383]
mannwhitneyu      p 0.074741
ks_1samp(a, norm.cdf)   p 0.3088
```

| question | parametric | nonparametric |
|---|---|---|
| two independent samples, same mean? | `ttest_ind` (`equal_var=False` for Welch) | `mannwhitneyu` |
| paired samples? | `ttest_rel` | `wilcoxon` |
| one sample vs a value? | `ttest_1samp` | `wilcoxon` |
| 3+ groups? | `f_oneway` | `kruskal` |
| does it follow this distribution? | `anderson`, `shapiro` | `ks_1samp`, `cramervonmises` |
| two samples, same distribution? | | `ks_2samp`, `epps_singleton_2samp` |
| categorical association? | `chi2_contingency`, `fisher_exact` | |
| correlation? | `pearsonr` | `spearmanr`, `kendalltau` |

Common keywords: `alternative=` (`'two-sided'`, `'less'`, `'greater'`), `nan_policy=`
(`'propagate'`, `'omit'`, `'raise'`), `axis=`. Most tests are vectorized over `axis`.

Multiple comparisons: `false_discovery_control(pvals, method='bh')` for Benjamini-Hochberg,
`dunnett` for many-vs-control, `tukey_hsd` for all-pairs after ANOVA.

`linregress(x, y)` gives slope, intercept, `rvalue`, `pvalue`, and standard errors in one
call for the simple 1-D case. For anything with more than one predictor use
`scipy.linalg.lstsq` (see `05-linalg-dense.md`) or `statsmodels` if you want inference.

## Resampling

When the sampling distribution of your statistic is unknown, resample instead of assuming.

```python
rng = np.random.default_rng(42)
rng.integers(0, 10, 5); rng.random(3); rng.normal(0, 1, 3); rng.choice(10, 4, replace=False)   # the draws shown above
a = rng.normal(0.0, 1, 50)          # the two samples used in the test table above
b = rng.normal(0.5, 1, 50)
res = stats.bootstrap((a,), np.median, n_resamples=2000, rng=np.random.default_rng(0))
print(res.confidence_interval, res.standard_error)
```

```
ConfidenceInterval(low=-0.3078, high=0.3688)   se 0.1713
```

`bootstrap` defaults to `method='BCa'` (bias-corrected and accelerated), which is the right
default; `'percentile'` and `'basic'` are the alternatives. `paired=True` for paired data.
Write the statistic to accept an `axis` argument: the default `vectorized=None` detects it
and evaluates every resample in one call (3 calls for 500 resamples, against 521 for a
statistic without `axis`). `vectorized=True` on a statistic without `axis` raises
`TypeError`.

`permutation_test` gives an exact or approximate p-value under the null of exchangeability,
with no distributional assumption at all:

```python
pt = stats.permutation_test(
    (a, b),
    lambda u, v, axis: np.mean(u, axis=axis) - np.mean(v, axis=axis),
    n_resamples=2000, rng=np.random.default_rng(0))
```

```
p = 0.08695652173913043
```

Compare to `ttest_ind`'s `0.0864`: close here because the data really is normal, and the
permutation test is the one you can defend when it is not. `permutation_type=` selects
`'independent'` (default, two-sample), `'samples'` (paired), or `'pairings'` (correlation).

`monte_carlo_test` is the version for a fully specified null distribution.

`scipy.stats.qmc` holds quasi-Monte Carlo sequences (`Sobol`, `Halton`, `LatinHypercube`)
with `discrepancy` to measure uniformity. Better than IID uniform sampling for filling a
design space, and what `integrate.qmc_quad` uses.

## `scipy.special`

Special functions and, more usefully day to day, the numerically stable versions of
expressions that overflow when written directly.

```
special.gamma(5)              24.0
special.erf(1)                0.842701
special.logsumexp([1000, 1001])  1001.313262      naive log(sum(exp(...))) -> inf
special.expit(0.5)            0.622459            = 1/(1+exp(-x))
special.softmax([1,2,3])      [0.090031 0.244728 0.665241]
special.roots_legendre(3)[0]  [-0.774597  0.  0.774597]
```

| need | call |
|---|---|
| `log(sum(exp(x)))` without overflow | `logsumexp` |
| logistic sigmoid and its inverse | `expit`, `logit` |
| softmax, log-softmax | `softmax`, `log_softmax` |
| `log(gamma(x))`, `log(n choose k)` | `gammaln`, `betaln`, `comb(..., exact=False)` |
| `log(1 + exp(x))` | `log1p(exp(x))` for small `x`, `logsumexp` generally |
| Bessel, Airy, elliptic, hypergeometric | `jv`, `yv`, `airy`, `ellipk`, `hyp2f1` |
| orthogonal polynomial roots and weights | `roots_legendre`, `roots_hermite`, `roots_laguerre` |
| error function family | `erf`, `erfc`, `erfinv`, `erfcx` |

`roots_legendre(n)` returning the Gauss-Legendre nodes and weights is the direct handle on
the quadrature rule from `08-integrate-quadrature.md`, for when you want the rule itself
rather than the integral.

## `scipy.spatial`

| need | call |
|---|---|
| nearest neighbors, radius queries | `cKDTree(points)`, then `.query(x, k=)`, `.query_ball_point` |
| all pairwise distances between two sets | `distance.cdist(A, B, metric='euclidean')` |
| condensed pairwise distances within one set | `distance.pdist(A)`, `distance.squareform` |
| triangulation, convex hull, Voronoi | `Delaunay`, `ConvexHull`, `Voronoi` |
| rotations, quaternions, slerp | `transform.Rotation`, `transform.Slerp` |

`cdist` supports about 20 metrics by name and is the fast path for a distance matrix.
`cKDTree.query` is `O(log n)` per query after an `O(n log n)` build, so it wins over `cdist`
as soon as you have more than a few thousand points and want only the nearest few.

## Gotchas

- **`1 - cdf(x)` in the far tail is zero.** Use `sf`. Same for `ppf(1-q)` versus `isf(q)`.
- **`fit` estimates `loc` by default.** For a distribution supported on `[0, inf)` that can
  produce a negative `loc` and biased shape estimates (with seed 0 the free `loc` came out
  `+0.09` instead). Pass `floc=0`.
- **`np.random.seed()` sets a global.** Two libraries both seeding it means neither is
  reproducible. Use `default_rng`.
- **`default_rng(root_seed + worker_id)` per worker is unsafe across runs.** Use
  `rng.spawn(k)`, or `default_rng([worker_id, root_seed])`.
- **`ttest_ind` assumes equal variances by default.** Pass `equal_var=False` (Welch) unless
  you have checked, which is almost always the better default.
- **A p-value is not an effect size.** Report `.confidence_interval()` alongside it; most
  tests now provide one.
- **Shape parameters come before `loc` and `scale`** in both `fit` output and positional
  calls. `gamma.fit` returning three numbers means `(a, loc, scale)`, not `(loc, scale, a)`.
- **A `bootstrap` statistic without an `axis` parameter is called about `n_resamples` times**
  in Python. Write it to accept `axis`; the default `vectorized=None` then batches it.
- **`rvs` without a seed is irreproducible.** Pass `random_state=rng` (distributions) or
  `rng=rng` (newer functions).
- **`nan` propagates silently through most reductions.** Set `nan_policy='omit'` or
  `'raise'` rather than discovering it in the output.
- **Frozen distributions are cheaper.** `d = stats.norm(2, 3); d.pdf(x)` avoids re-validating
  parameters on every call inside a loop.

## Quick reference

| task | call |
|---|---|
| seeded generator | `rng = np.random.default_rng(seed)` |
| independent parallel streams | `children = rng.spawn(n_workers)` |
| uniform / normal / integers | `rng.random(n)`, `rng.normal(mu, sd, n)`, `rng.integers(lo, hi, n)` |
| sample without replacement | `rng.choice(a, k, replace=False)` |
| shuffle in place | `rng.shuffle(a)` |
| frozen distribution | `d = stats.norm(loc=, scale=)` |
| upper tail probability | `d.sf(x)` |
| quantile | `d.ppf(q)` / `d.isf(q)` |
| MLE fit with fixed location | `stats.gamma.fit(data, floc=0)` |
| modern fit with bounds | `stats.fit(stats.gamma, data, bounds=...)` |
| Welch two-sample test | `stats.ttest_ind(a, b, equal_var=False)` |
| nonparametric two-sample | `stats.mannwhitneyu(a, b)` |
| goodness of fit | `stats.ks_1samp(x, dist.cdf)`, `stats.anderson(x)` |
| correlation | `stats.pearsonr`, `stats.spearmanr` |
| simple regression | `stats.linregress(x, y)` |
| multiple-comparison control | `stats.false_discovery_control(pvals)` |
| CI without assumptions | `stats.bootstrap((x,), stat, rng=rng)` with `stat(x, axis)` |
| p-value without assumptions | `stats.permutation_test((a, b), stat, rng=rng)` |
| space-filling design | `stats.qmc.Sobol(d, rng=rng).random(n)` |
| stable log-sum-exp | `special.logsumexp(x, axis=)` |
| Gauss-Legendre rule | `special.roots_legendre(n)` |
| nearest neighbors | `spatial.cKDTree(pts).query(q, k=5)` |
| distance matrix | `spatial.distance.cdist(A, B)` |
