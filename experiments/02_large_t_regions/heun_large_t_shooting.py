
"""
Large-t Heun shooting and outer/inner hypergeometric regions
=============================================================

Numerical realization of the generic Heun problem in Sec. 2 of
arXiv:2608.00363, using our paper's Schrödinger-form convention

  Psi''(z) + Q(z) Psi(z) = 0

with singularities z = 0, 1, t, infinity.

For a clean real numerical demonstration we choose

    a0 = 1
    a1 = 0.2
    at = 0.3
    ainf = 1
    t = 1000

and a one-parameter spectral slice

    u(omega) = -omega,

with all a_i fixed.  In a physical Kerr-(A)dS application all a_i and u
are functions of omega; here we isolate the large-t Heun mechanism itself.

To avoid crossing the finite singularities z=1 and z=t for real positive t,
the numerical contour is the negative real ray

    z = -x,  x > 0.

Nothing about the large-t patch structure changes: the asymptotic conditions
are naturally expressed in |z|=x.

Conventions from our paper used:
  full Heun potential: eq. (2.1)
  outer hypergeometric potential: eqs. (2.4)-(2.5)
  inner coordinate zeta=z/t: eq. (2.48)
  inner hypergeometric potential: eqs. (2.51)-(2.53)
  matching region: eq. (2.56), 1 << z << t

We impose Frobenius boundary conditions
    Psi ~ x^(1/2+a0)        at x -> 0
    Psi ~ x^(1/2-ainf)      at x -> infinity

and shoot from both endpoints to x_match=sqrt(t).

The script writes:
  spectrum_condition.png/pdf
  eigenfunctions.png/pdf
  region_split.png/pdf
  mode_matching.png/pdf
  eigenvalues.csv
  mismatch_scan.csv
  region_errors.csv
  selected_mode.csv
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class HeunParameters:
    a0: float = 1.0
    a1: float = 0.2
    at: float = 0.3
    ainf: float = 1.0
    t: float = 1000.0

P = HeunParameters()

EPS = 1.0e-7
XMAX_FACTOR = 100.0
XMAX = P.t * XMAX_FACTOR
X_MATCH = np.sqrt(P.t)

SCAN_RTOL = 3.0e-8
PROFILE_RTOL = 1.0e-10
ATOL = 1.0e-11

OMEGA_MIN = 1.2
OMEGA_MAX = 12.0
N_SCAN = 190
N_MODES_PLOT = 6

OUTDIR = Path(__file__).resolve().parent / "output"
OUTDIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Full Heun and the two leading hypergeometric operators
# ---------------------------------------------------------------------

def u_of_omega(omega):
    """One-parameter spectral slice used in this numerical demonstration."""
    return -omega


def Q_full(z, omega, p=P):
    """Full Heun potential, paper eq. (2.1)."""
    u = u_of_omega(omega)
    C = 0.5 - p.a1**2 - p.at**2 - p.a0**2 + p.ainf**2 + u

    return (
        (0.25 - p.a0**2) / z**2
        + (0.25 - p.a1**2) / (z - 1.0)**2
        + (0.25 - p.at**2) / (z - p.t)**2
        - C / (z * (z - 1.0))
        + u / (z * (z - p.t))
    )


def lambda_out(omega, p=P):
    """Paper eq. (2.5)."""
    u = u_of_omega(omega)
    return -p.at**2 + p.ainf**2 - p.a0**2 - p.a1**2 + u + 0.5


def Q_outer(z, omega, p=P):
    """Leading outer hypergeometric potential, paper eq. (2.4)."""
    lam = lambda_out(omega, p)
    return (
        -lam / ((z - 1.0) * z)
        + (0.25 - p.a0**2) / z**2
        + (0.25 - p.a1**2) / (z - 1.0)**2
    )


def lambda_in(omega, p=P):
    """Paper eq. (2.53): -lambda_in = u."""
    return -u_of_omega(omega)


def Q_inner(zeta, omega, p=P):
    """Leading inner hypergeometric potential in zeta=z/t, eq. (2.52)."""
    u = u_of_omega(omega)
    lam = lambda_in(omega, p)

    return (
        -lam / ((zeta - 1.0) * zeta)
        + (0.25 - p.at**2) / (zeta - 1.0)**2
        + (p.at**2 - p.ainf**2 - u) / zeta**2
    )


def kappa_squared(omega, p=P):
    """From eqs. (2.17)-(2.18):
       kappa^2 = 1/4 + ainf^2 - at^2 + u0.
       Here u=u0=-omega.
    """
    return 0.25 + p.ainf**2 - p.at**2 + u_of_omega(omega)


# ---------------------------------------------------------------------
# Integrate in s = log x on the ray z=-x.
#
# If y(x)=Psi(-x), then
#   y_xx + Q(-x)y = 0.
# In s=log x,
#   y_ss - y_s + x^2 Q(-x)y = 0.
# ---------------------------------------------------------------------

def q_in_x(x, omega, p=P):
    """Inner leading operator expressed back in the original x=|z| variable."""
    zeta = -x / p.t
    return Q_inner(zeta, omega, p) / p.t**2


def rhs_logx(s, Y, omega, model="full", p=P):
    x = np.exp(s)

    if model == "full":
        q = Q_full(-x, omega, p)
    elif model == "outer":
        q = Q_outer(-x, omega, p)
    elif model == "inner":
        q = q_in_x(x, omega, p)
    else:
        raise ValueError(model)

    return np.array([Y[1], Y[1] - x*x*q*Y[0]])


def integrate_branch(
    omega,
    model,
    side,
    p=P,
    xmatch=X_MATCH,
    rtol=PROFILE_RTOL,
    dense_output=False,
):
    """Integrate an endpoint-selected Frobenius branch to xmatch."""
    sm = np.log(xmatch)

    if side == "left":
        s0 = np.log(EPS)
        # amplitude arbitrary; only the Frobenius logarithmic derivative matters
        y0 = np.array([1.0, 0.5 + p.a0])
        span = (s0, sm)

    elif side == "right":
        s0 = np.log(p.t * XMAX_FACTOR)
        y0 = np.array([1.0, 0.5 - p.ainf])
        span = (s0, sm)

    else:
        raise ValueError(side)

    sol = solve_ivp(
        lambda s, y: rhs_logx(s, y, omega, model, p),
        span,
        y0,
        method="DOP853",
        rtol=rtol,
        atol=ATOL,
        dense_output=dense_output,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol


def mismatch(omega, p=P):
    """Normalized Wronskian of the two full-Heun endpoint solutions."""
    L = integrate_branch(
        omega, "full", "left", p, rtol=SCAN_RTOL
    ).y[:, -1]

    R = integrate_branch(
        omega, "full", "right", p, rtol=SCAN_RTOL
    ).y[:, -1]

    det = L[0]*R[1] - L[1]*R[0]
    return det / (np.linalg.norm(L) * np.linalg.norm(R))


# ---------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------

def find_spectrum():
    grid = np.linspace(OMEGA_MIN, OMEGA_MAX, N_SCAN)
    vals = np.array([mismatch(w) for w in grid])

    roots = []
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if fa == 0.0:
            roots.append(a)
        elif fa * fb < 0.0:
            root = brentq(
                mismatch, a, b,
                xtol=1e-11, rtol=1e-11, maxiter=100
            )
            if not roots or abs(root - roots[-1]) > 1e-7:
                roots.append(root)

    return grid, vals, np.asarray(roots)


# ---------------------------------------------------------------------
# Full eigenfunction and leading outer/inner approximations
# ---------------------------------------------------------------------

def full_and_patch_profiles(omega, x):
    """Return global full-Heun mode plus the two endpoint-normalized
    leading hypergeometric approximations.

    The outer approximation uses exactly the same left Frobenius
    normalization as the full solution.
    The inner approximation uses exactly the same right Frobenius
    normalization, including the global scale used to join the full
    left and right shots.
    """
    full_L = integrate_branch(
        omega, "full", "left", dense_output=True
    )
    full_R = integrate_branch(
        omega, "full", "right", dense_output=True
    )
    out_L = integrate_branch(
        omega, "outer", "left", dense_output=True
    )
    in_R = integrate_branch(
        omega, "inner", "right", dense_output=True
    )

    Lm = full_L.y[:, -1]
    Rm = full_R.y[:, -1]
    right_scale = np.dot(Lm, Rm) / np.dot(Rm, Rm)

    s = np.log(x)
    mask = x <= X_MATCH

    y_full = np.empty_like(x)
    y_full[mask] = full_L.sol(s[mask])[0]
    y_full[~mask] = right_scale * full_R.sol(s[~mask])[0]

    y_outer = out_L.sol(s)[0]
    y_inner = right_scale * in_R.sol(s)[0]

    norm = np.max(np.abs(y_full))
    return y_full/norm, y_outer/norm, y_inner/norm


def full_profile(omega, x):
    y, _, _ = full_and_patch_profiles(omega, x)
    return y


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("Parameters:", P)
    print("Contour: z=-x, x>0")
    print("Spectral slice: u(omega)=-omega")
    print("Match point x=sqrt(t) =", X_MATCH)

    grid, D, roots = find_spectrum()

    print("\nEigenvalues:")
    for n, w in enumerate(roots):
        k2 = kappa_squared(w)
        print(
            f"n={n:2d}  omega={w:.12f}  "
            f"u={-w:.12f}  kappa^2={k2:.12f}  "
            f"D={mismatch(w):+.2e}"
        )

    # -------------------------------------------------------------
    # CSV: spectrum
    # -------------------------------------------------------------
    with (OUTDIR / "eigenvalues.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["n", "omega", "u=-omega", "kappa_squared", "mismatch"])
        for n, w in enumerate(roots):
            wr.writerow([
                n,
                f"{w:.15g}",
                f"{-w:.15g}",
                f"{kappa_squared(w):.15g}",
                f"{mismatch(w):.6e}",
            ])

    with (OUTDIR / "mismatch_scan.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["omega", "D(omega)"])
        wr.writerows(zip(grid, D))

    # -------------------------------------------------------------
    # Plot: spectrum condition
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(grid, D, label=r"$D(\omega)$")
    ax.axhline(0.0, linewidth=0.8)
    ax.scatter(roots, np.zeros_like(roots), zorder=3, label="roots")
    ax.set_xlabel(r"$\omega$  (with $u=-\omega$)")
    ax.set_ylabel(r"normalized Wronskian $D(\omega)$")
    ax.set_title(fr"Full Heun shooting spectrum, $t={P.t:g}$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "spectrum_condition.png", dpi=180)
    fig.savefig(OUTDIR / "spectrum_condition.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Region split in the original variable x=|z|
    # -------------------------------------------------------------
    selected = roots[0]

    xerr = np.logspace(-3, 6, 1600)
    qf = np.array([Q_full(-xx, selected) for xx in xerr])
    qo = np.array([Q_outer(-xx, selected) for xx in xerr])
    qi_scaled = np.array([
        Q_inner(-xx/P.t, selected) for xx in xerr
    ])
    # Compare t^2 Q with Q_in, as in paper eq. (2.51)
    tqf = P.t**2 * qf

    Rout = np.abs(qf - qo) / (np.abs(qo) + 1e-300)
    Rin = np.abs(tqf - qi_scaled) / (np.abs(qi_scaled) + 1e-300)

    with (OUTDIR / "region_errors.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "x_abs_z",
            "abs_Qint_outer_over_Q0_outer",
            "abs_Qint_inner_over_Q0_inner",
        ])
        wr.writerows(zip(xerr, Rout, Rin))

    # A concrete way to visualize << is to choose delta=0.1.
    # Then the nominal outer window is x/t < delta,
    # the nominal inner window is 1/x < delta,
    # and both hold for 1/delta < x < delta*t.
    delta = 0.1
    overlap_left = 1.0 / delta
    overlap_right = delta * P.t

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.loglog(
        xerr, Rout,
        label=r"outer: $|Q-Q^{\rm out}_0|/|Q^{\rm out}_0|$"
    )
    ax.loglog(
        xerr, Rin,
        label=r"inner: $|t^2Q-Q^{\rm in}_0|/|Q^{\rm in}_0|$"
    )
    ax.axhline(delta, linewidth=0.8)
    ax.axvline(1.0, linewidth=0.8)
    ax.axvline(np.sqrt(P.t), linewidth=0.8)
    ax.axvline(P.t, linewidth=0.8)
    ax.axvspan(
        overlap_left, overlap_right, alpha=0.12,
        label=fr"illustrative overlap: ${overlap_left:g}<|z|<{overlap_right:g}$"
    )
    ax.set_xlabel(r"$x=|z|$ on the ray $z=-x$")
    ax.set_ylabel("relative interaction size")
    ax.set_title(
        fr"Outer/inner split of the full Heun operator, $t={P.t:g}$"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUTDIR / "region_split.png", dpi=180)
    fig.savefig(OUTDIR / "region_split.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot several exact eigenfunctions
    # -------------------------------------------------------------
    xprof = np.logspace(-5, 5, 1500)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    nplot = min(N_MODES_PLOT, len(roots))

    modes = []
    for n, w in enumerate(roots[:nplot]):
        y = full_profile(w, xprof)
        modes.append(y)
        ax.semilogx(
            xprof, y,
            label=fr"$n={n}$, $\omega={w:.4f}$"
        )

    ax.axvspan(overlap_left, overlap_right, alpha=0.10)
    ax.set_xlabel(r"$x=|z|$")
    ax.set_ylabel(r"normalized $\Psi_n$")
    ax.set_title(
        fr"Full Heun eigenfunctions from Frobenius--Frobenius shooting, $t={P.t:g}$"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "eigenfunctions.png", dpi=180)
    fig.savefig(OUTDIR / "eigenfunctions.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Full mode vs the two leading hypergeometric patches
    # -------------------------------------------------------------
    yfull, yout, yin = full_and_patch_profiles(selected, xprof)

    with (OUTDIR / "selected_mode.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "x_abs_z",
            "Psi_full",
            "Psi_outer_leading",
            "Psi_inner_leading",
        ])
        wr.writerows(zip(xprof, yfull, yout, yin))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.semilogx(
        xprof, yfull,
        linewidth=2.0,
        label=fr"full Heun, $\omega={selected:.6f}$"
    )

    mask_out = xprof <= P.t
    mask_in = xprof >= 1.0

    ax.semilogx(
        xprof[mask_out], yout[mask_out],
        label="leading outer hypergeometric"
    )
    ax.semilogx(
        xprof[mask_in], yin[mask_in],
        label="leading inner hypergeometric"
    )
    ax.axvspan(
        overlap_left, overlap_right, alpha=0.12,
        label=r"$1\ll |z|\ll t$ overlap"
    )
    ax.axvline(np.sqrt(P.t), linewidth=0.8)
    ax.set_xlabel(r"$x=|z|$")
    ax.set_ylabel(r"common normalized wavefunction")
    ax.set_title(
        "One full Heun eigenmode resolved into the two large-t patches"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUTDIR / "mode_matching.png", dpi=180)
    fig.savefig(OUTDIR / "mode_matching.pdf")
    plt.close(fig)

    print("\nFor the first eigenmode, operator errors near the matching scale:")
    for xx in [1, 3, 10, 30, 100, 300, 1000]:
        j = np.argmin(np.abs(xerr - xx))
        print(
            f"x={xx:6g}: outer={Rout[j]:.4g}, inner={Rin[j]:.4g}"
        )

    print("\nWrote:")
    for pth in sorted(OUTDIR.iterdir()):
        print(" ", pth.name)


if __name__ == "__main__":
    main()
