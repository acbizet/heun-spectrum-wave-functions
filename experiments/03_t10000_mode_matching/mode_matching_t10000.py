
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

@dataclass(frozen=True)
class HeunParameters:
    a0: float = 1.0
    a1: float = 0.2
    at: float = 0.3
    ainf: float = 1.0
    t: float = 10000.0

P = HeunParameters()

EPS = 1.0e-7
XMAX_FACTOR = 100.0
X_MATCH = np.sqrt(P.t)

SCAN_RTOL = 5.0e-8
PROFILE_RTOL = 1.0e-10
ATOL = 1.0e-11

# Search range chosen to capture several low modes at t=10000
OMEGA_MIN = 1.0
OMEGA_MAX = 8.0
N_SCAN = 220

# We want "another mode" relative to the previous n=0 plot.
TARGET_MODE_INDEX = 1

OUTDIR = Path(__file__).resolve().parent / "output"
OUTDIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Heun full operator and large-t approximations
# ------------------------------------------------------------

def u_of_omega(omega):
    return -omega

def Q_full(z, omega, p=P):
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
    u = u_of_omega(omega)
    return -p.at**2 + p.ainf**2 - p.a0**2 - p.a1**2 + u + 0.5

def Q_outer(z, omega, p=P):
    lam = lambda_out(omega, p)
    return (
        -lam / (z * (z - 1.0))
        + (0.25 - p.a0**2) / z**2
        + (0.25 - p.a1**2) / (z - 1.0)**2
    )

def lambda_in(omega, p=P):
    return -u_of_omega(omega)

def Q_inner(zeta, omega, p=P):
    u = u_of_omega(omega)
    lam = lambda_in(omega, p)
    return (
        -lam / (zeta * (zeta - 1.0))
        + (0.25 - p.at**2) / (zeta - 1.0)**2
        + (p.at**2 - p.ainf**2 - u) / zeta**2
    )

def q_in_x(x, omega, p=P):
    zeta = -x / p.t
    return Q_inner(zeta, omega, p) / p.t**2

# ------------------------------------------------------------
# ODE in s = log x, x = |z| on the ray z=-x
# ------------------------------------------------------------

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
    omega, model, side, p=P, xmatch=X_MATCH,
    rtol=PROFILE_RTOL, dense_output=False
):
    sm = np.log(xmatch)
    if side == "left":
        s0 = np.log(EPS)
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
        span, y0,
        method="DOP853",
        rtol=rtol, atol=ATOL,
        dense_output=dense_output
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol

def mismatch(omega, p=P):
    L = integrate_branch(omega, "full", "left", p, rtol=SCAN_RTOL).y[:, -1]
    R = integrate_branch(omega, "full", "right", p, rtol=SCAN_RTOL).y[:, -1]
    det = L[0]*R[1] - L[1]*R[0]
    return det / (np.linalg.norm(L) * np.linalg.norm(R))

def find_spectrum():
    grid = np.linspace(OMEGA_MIN, OMEGA_MAX, N_SCAN)
    vals = np.array([mismatch(w) for w in grid])
    roots = []
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if fa == 0.0:
            roots.append(a)
        elif fa * fb < 0.0:
            root = brentq(mismatch, a, b, xtol=1e-11, rtol=1e-11, maxiter=100)
            if not roots or abs(root - roots[-1]) > 1e-7:
                roots.append(root)
    return grid, vals, np.asarray(roots)

# ------------------------------------------------------------
# Profiles
# ------------------------------------------------------------

def full_and_patch_profiles(omega, x):
    full_L = integrate_branch(omega, "full", "left", dense_output=True)
    full_R = integrate_branch(omega, "full", "right", dense_output=True)
    out_L = integrate_branch(omega, "outer", "left", dense_output=True)
    in_R = integrate_branch(omega, "inner", "right", dense_output=True)

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

def main():
    print("Parameters:", P)
    print("Using t =", P.t)
    print("Match point x = sqrt(t) =", X_MATCH)
    print("Target mode index =", TARGET_MODE_INDEX)

    grid, D, roots = find_spectrum()
    print("\nEigenvalues found in scan window:")
    for n, w in enumerate(roots):
        print(f"n={n:2d}   omega={w:.12f}   D={mismatch(w):+.3e}")

    if len(roots) <= TARGET_MODE_INDEX:
        raise RuntimeError("Did not find enough modes in the scan range.")

    omega = roots[TARGET_MODE_INDEX]
    print(f"\nSelected mode n={TARGET_MODE_INDEX} with omega={omega:.12f}")

    x = np.logspace(-5, 6, 2200)
    yfull, yout, yin = full_and_patch_profiles(omega, x)

    delta = 0.1
    overlap_left = 1.0 / delta
    overlap_right = delta * P.t

    # Save spectrum
    with (OUTDIR / "eigenvalues_t10000.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["n", "omega", "mismatch"])
        for n, w in enumerate(roots):
            wr.writerow([n, f"{w:.15g}", f"{mismatch(w):.6e}"])

    # Save selected mode data
    with (OUTDIR / "mode_matching_t10000_n1.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["x_abs_z", "Psi_full", "Psi_outer_leading", "Psi_inner_leading"])
        wr.writerows(zip(x, yfull, yout, yin))

    # Plot mode matching
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.semilogx(x, yfull, linewidth=2.1, label=fr"full Heun, mode $n={TARGET_MODE_INDEX}$, $\omega={omega:.6f}$")

    mask_out = x <= P.t
    mask_in = x >= 1.0
    ax.semilogx(x[mask_out], yout[mask_out], label="leading outer hypergeometric")
    ax.semilogx(x[mask_in], yin[mask_in], label="leading inner hypergeometric")

    ax.axvspan(overlap_left, overlap_right, alpha=0.12, label=fr"illustrative overlap: ${overlap_left:g}<|z|<{overlap_right:g}$")
    ax.axvline(X_MATCH, linewidth=0.9, label=fr"$|z|_{{match}}=\sqrt{{t}}={X_MATCH:.0f}$")

    ax.set_xlabel(r"$x=|z|$ on the contour $z=-x$")
    ax.set_ylabel(r"common normalized wavefunction")
    ax.set_title(fr"Mode matching for the full Heun eigenmode at $t={P.t:g}$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUTDIR / "mode_matching_t10000_n1.png", dpi=180)
    fig.savefig(OUTDIR / "mode_matching_t10000_n1.pdf")
    plt.close(fig)

    # Also save spectrum condition for context
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(grid, D, label=r"$D(\omega)$")
    ax.axhline(0.0, linewidth=0.8)
    ax.scatter(roots, np.zeros_like(roots), zorder=3, label="roots")
    ax.scatter([omega], [0.0], zorder=4, s=60, label=fr"selected mode $n={TARGET_MODE_INDEX}$")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"normalized Wronskian $D(\omega)$")
    ax.set_title(fr"Shooting spectrum at $t={P.t:g}$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUTDIR / "spectrum_t10000.png", dpi=180)
    plt.close(fig)

    print("\nWrote:")
    for p in sorted(OUTDIR.iterdir()):
        print(" ", p.name)

if __name__ == "__main__":
    main()
