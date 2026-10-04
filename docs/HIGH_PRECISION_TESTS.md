# Large-`t` high-precision numerical tests

## 1. What is being tested

The numerical program compares the full four-singularity Heun problem against the two complementary Gauss-hypergeometric descriptions we propose in arXiv:2608.00363.  The full equation is solved by two-sided Frobenius shooting.  In the large-`t` problem the original variable naturally separates into

- an **outer patch**, valid for `|z|/t << 1`,
- an **inner patch**, obtained at fixed `ζ=z/t` and valid for `|z| >> 1`,
- an **overlap region**, `1 << |z| << t`.

The use of `|z|` here reflects the numerical contour `z=-x`, `x>0`, chosen to avoid the finite singularities at `z=1` and `z=t` in the real demonstration.

The matching scale `|z| ~ sqrt(t)` is convenient because it is logarithmically central in the overlap, but it is not a physical boundary.

## 2. Sequence of computations in the repository

### Experiment 01 — baseline shooting

`experiments/01_basic_shooting/`

A first real Sturm-Liouville-like spectral slice with `t=2` and `u=-ω`.  It establishes the numerical machinery: Frobenius data, logarithmic radial coordinate, two-sided shooting, normalized-Wronskian spectral condition, root finding, and eigenfunction reconstruction.

This is a methodological baseline rather than a large-`t` or astrophysical calculation.

### Experiment 02 — resolving the two large-`t` regions

`experiments/02_large_t_regions/`

Parameters:

```text
a0=1, a1=0.2, at=0.3, ainf=1, t=1000, u=-ω.
```

The full operator is compared directly with the leading outer and inner hypergeometric operators.  The resulting `region_split.png` makes the nonuniform large-`t` limit visible in the original coordinate.

For the first numerical eigenmode, representative relative interaction sizes were

| `|z|` | outer relative interaction | inner relative interaction |
|---:|---:|---:|
| 1 | 9.96e-3 | 1.27 |
| 10 | 3.44e-2 | 2.49e-1 |
| 30 | 8.31e-2 | 9.48e-2 |
| 100 | 2.42e-1 | 3.52e-2 |
| 1000 | 1.24 | 1.13e-2 |

Thus the outer description dominates near the `z=0` end, the inner description dominates toward the `z~t`/large-`|z|` side, and both are usable around the central overlap.

### Experiment 03 — wider overlap at `t=10000`

`experiments/03_t10000_mode_matching/`

The same illustrative `u=-ω` family is pushed to `t=10000`; the second low mode (`n=1`) is shown.  The natural central matching scale moves to `sqrt(t)=100`, and the overlap becomes very broad.

This computation is useful to visualize the asymptotic patch structure, but **it is not the right family on which to test our explicit first diffeomorphism correction**, because those formulae assume a regular `u(t)=u0+u1/t+...` relation fixed by the isospectral construction.

### Experiment 04 — paper-consistent first correction

`experiments/04_first_correction/`

This is the central precision test.  We use the expansion of the accessory parameter in eqs. (3.3)-(3.4),

```text
u = u0(kappa) + u1(kappa)/t,
```

and the first outer/inner diffeomorphisms of eqs. (3.21) and (3.23)-(3.24).  The reconstructed wave function uses the inverse-diffeomorphism action of eq. (3.41).

At `t=10000`, selecting the second low mode gives

```text
kappa^2 = -0.556874971371
u0      = -1.716874971371
u1/t    = -1.87979e-4
u        = -1.717062949900
```

The first-order polynomial coefficients are

```text
A12 = -1.063903970434
A11 =  1.063903970434
B11 = -1.094887705074
B10 =  1.094887705074
```

The strongest numerical result is the reduction of the full-Heun wave-function error after adding the `O(1/t)` reconstruction.  At the central scale `|z|=sqrt(t)=100`:

| patch | leading absolute error | corrected absolute error | improvement |
|---|---:|---:|---:|
| outer | 1.666e-3 | 2.269e-6 | ~7.34e2 |
| inner | 1.687e-3 | 2.560e-6 | ~6.59e2 |

Further into the overlap/inner region, e.g. `|z|=300`, the inner error changes from `1.216e-3` to `5.402e-7`, an improvement of roughly `2.25e3`.

The correction is not uniformly helpful outside the intended domain of a given patch; for example, the inner approximation is not expected to improve near `|z|~1`.  The precision statement is therefore a **matched-asymptotic** one, not a claim of a single uniform local expansion over the entire domain.

## 3. Why the result is significant

In our paper, we explicitly propose numerically testing whether the `O(1/t)` wave-function corrections enforce a smooth transition between inner and outer hypergeometric patches at finite `t`.  The fourth experiment is a direct realization of this proposed check in a generic real Heun model.  The reduction from errors of order `10^-3` to order `10^-6` near `sqrt(t)` is strong numerical evidence that the first diffeomorphism correction captures the dominant finite-`t` deformation of the mode shape in the overlap.

The test also separates two logically different questions:

1. **local shape accuracy** — does a leading/corrected hypergeometric patch reproduce the full Heun eigenfunction when evaluated at the correct spectral data?;
2. **spectral prediction** — can the hypergeometric matching/Robin condition itself predict the spectral data without first solving the full Heun problem?

The current repository establishes (1) very accurately.  A physically complete next stage should test (2) with the Kerr-(A)dS parameter map and complex frequency.

## 4. Precision roadmap

The most useful next convergence study is to repeat the fourth experiment for a sequence such as

```text
t = 10^2, 10^3, 10^4, 10^5
```

at a fixed mode label and evaluate errors at `|z|=sqrt(t)` and at several fixed overlap fractions.  Fitting

```text
E_leading(t)   ~ t^{-alpha}
E_corrected(t) ~ t^{-beta}
```

would convert the current high-precision spot check into a direct numerical measurement of the asymptotic power counting.
