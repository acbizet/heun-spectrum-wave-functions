
"""
Numerical shooting for the Schrödinger-form Heun equation used in
arXiv:2608.00363, eq. (2.1).

We choose a one-parameter toy spectral embedding
    u(omega) = -omega
with fixed
    t=2, a0=1, a1=0.2, at=0.3, ainf=1.

To avoid crossing the finite singularities z=1 and z=t=2, shoot on the
negative real ray
    z = -x,   x in (0, infinity).

The selected Frobenius branches are
    Psi ~ x^(1/2+a0) = x^(3/2)       as x -> 0,
    Psi ~ x^(1/2-ainf) = x^(-1/2)    as x -> infinity.

For these parameters the equation is
    y''(x) + [V0(x) + omega W(x)] y(x) = 0
with W(x)>0, so omega plays the role of a generalized Sturm-Liouville
eigenvalue.

The code:
  * builds high-order Frobenius initial data at both endpoints,
  * integrates from both ends to x_match=1,
  * uses a normalized Wronskian as the eigenvalue condition,
  * scans and refines roots with Brent's method,
  * plots the shooting mismatch and several eigenfunctions,
  * writes eigenvalues and sampled eigenfunctions to CSV files.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ----------------------- parameters and equation ----------------------- #

@dataclass(frozen=True)
class HeunParams:
    t: float = 2.0
    a0: float = 1.0
    a1: float = 0.2
    at: float = 0.3
    ainf: float = 1.0

P = HeunParams()

# Numerical controls
EPS = 1.0e-6
XMAX = 1.0e6
X_MATCH = 1.0
FROB_ORDER = 8
RTOL = 2.0e-9
ATOL = 1.0e-11

# Spectral scan
OMEGA_MIN = 1.0
OMEGA_MAX = 280.0
N_SCAN = 900
N_MODES_TO_PLOT = 6

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
OUTDIR = HERE / "output"
OUTDIR.mkdir(exist_ok=True)

def u_of_omega(omega: float) -> float:
    """Toy frequency dependence, consistent with our paper's observation that
    Heun parameters are functions of the physical frequency omega."""
    return -omega

def heun_Q(z, omega, p=P):
    """Q(z) in our paper's Schrödinger-form Heun equation."""
    u = u_of_omega(omega)
    C = 0.5 - p.a1**2 - p.at**2 - p.a0**2 + p.ainf**2 + u
    return (
        (0.25 - p.a0**2) / z**2
        + (0.25 - p.a1**2) / (z - 1.0)**2
        + (0.25 - p.at**2) / (z - p.t)**2
        - C / (z * (z - 1.0))
        + u / (z * (z - p.t))
    )

def Vx(x, omega, p=P):
    """Potential on the negative real ray z=-x."""
    # Written explicitly to keep the arithmetic real.
    u = u_of_omega(omega)
    C = 0.5 - p.a1**2 - p.at**2 - p.a0**2 + p.ainf**2 + u
    return (
        (0.25 - p.a0**2) / x**2
        + (0.25 - p.a1**2) / (x + 1.0)**2
        + (0.25 - p.at**2) / (x + p.t)**2
        - C / (x * (x + 1.0))
        + u / (x * (x + p.t))
    )

def spectral_weight(x, p=P):
    """Coefficient W(x) multiplying omega when u=-omega."""
    return (p.t - 1.0) / (x * (x + 1.0) * (x + p.t))

# ----------------------- Frobenius expansions ----------------------- #

def zero_potential_coeffs(omega, max_power, p=P):
    """Laurent coefficients q_j in V(x)=sum_{j=-2}^\infty q_j x^j near x=0."""
    u = u_of_omega(omega)
    C = 0.5 - p.a1**2 - p.at**2 - p.a0**2 + p.ainf**2 + u

    q = {-2: 0.25 - p.a0**2}
    q[-1] = -C + u / p.t

    B1 = 0.25 - p.a1**2
    Bt = 0.25 - p.at**2

    for j in range(max_power + 1):
        # B1/(1+x)^2
        val = B1 * (-1)**j * (j + 1)
        # Bt/(t+x)^2
        val += Bt * (-1)**j * (j + 1) / p.t**(j + 2)
        # -C/[x(1+x)]
        val += -C * (-1)**(j + 1)
        # +u/[x(t+x)]
        val += u * (-1)**(j + 1) / p.t**(j + 2)
        q[j] = val
    return q

def infinity_potential_coeffs(omega, max_k, p=P):
    """Coefficients v_k in V(x)=sum_{k=2}^\infty v_k x^{-k} near infinity."""
    u = u_of_omega(omega)
    C = 0.5 - p.a1**2 - p.at**2 - p.a0**2 + p.ainf**2 + u

    A0 = 0.25 - p.a0**2
    B1 = 0.25 - p.a1**2
    Bt = 0.25 - p.at**2

    v = {}
    for k in range(2, max_k + 1):
        n = k - 2
        val = A0 if k == 2 else 0.0
        val += B1 * (-1)**n * (n + 1)
        val += Bt * (-1)**n * (n + 1) * p.t**n
        val += -C * (-1)**n
        val += u * (-1)**n * p.t**n
        v[k] = val
    return v

def frobenius_state_zero(eps, omega, order=FROB_ORDER, p=P):
    """[y, dy/ds] at x=eps, s=log x, for the subleading x^(1/2+a0) branch."""
    exponent = 0.5 + p.a0
    q = zero_potential_coeffs(omega, order - 2, p)

    c = np.zeros(order + 1)
    c[0] = 1.0

    for n in range(1, order + 1):
        denom = (exponent + n) * (exponent + n - 1.0) + q[-2]
        source = 0.0
        for j in range(-1, n - 1):  # j=-1,...,n-2
            source += q[j] * c[n - 2 - j]
        c[n] = -source / denom

    powers = float(eps) ** np.arange(order + 1)
    y = eps**exponent * np.dot(c, powers)
    # dy/ds = x dy/dx
    ys = eps**exponent * np.dot((exponent + np.arange(order + 1)) * c, powers)
    return np.array([y, ys], dtype=float)

def frobenius_state_infinity(xmax, omega, order=FROB_ORDER, p=P):
    """[y, dy/ds] at x=xmax for the decaying x^(1/2-ainf) branch."""
    exponent = 0.5 - p.ainf
    v = infinity_potential_coeffs(omega, order + 2, p)

    d = np.zeros(order + 1)
    d[0] = 1.0

    for n in range(1, order + 1):
        denom = (exponent - n) * (exponent - n - 1.0) + v[2]
        source = 0.0
        for k in range(3, n + 3):  # k=3,...,n+2
            source += v[k] * d[n + 2 - k]
        d[n] = -source / denom

    invpowers = float(xmax) ** (-np.arange(order + 1))
    y = xmax**exponent * np.dot(d, invpowers)
    ys = xmax**exponent * np.dot((exponent - np.arange(order + 1)) * d, invpowers)
    return np.array([y, ys], dtype=float)

# ----------------------- shooting in s = log x ----------------------- #

def rhs_logx(s, Y, omega, p=P):
    """Equation in s=log x:
       Y_ss - Y_s + x^2 V(x) Y = 0.
    """
    x = np.exp(s)
    return np.array([Y[1], Y[1] - x*x*Vx(x, omega, p)*Y[0]])

def integrate_to_match(omega, side, p=P):
    sm = np.log(X_MATCH)
    if side == "left":
        s0 = np.log(EPS)
        y0 = frobenius_state_zero(EPS, omega, p=p)
        span = (s0, sm)
    elif side == "right":
        s0 = np.log(XMAX)
        y0 = frobenius_state_infinity(XMAX, omega, p=p)
        span = (s0, sm)
    else:
        raise ValueError("side must be 'left' or 'right'")

    sol = solve_ivp(
        lambda s, y: rhs_logx(s, y, omega, p),
        span,
        y0,
        method="DOP853",
        rtol=RTOL,
        atol=ATOL,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1]

def mismatch(omega, p=P):
    """Normalized Wronskian at x=X_MATCH.

    The root condition is det([Y_L,Y_R])=0, i.e. the two endpoint-selected
    solutions are linearly dependent and therefore form one global mode.
    """
    L = integrate_to_match(omega, "left", p)
    R = integrate_to_match(omega, "right", p)
    det = L[0]*R[1] - L[1]*R[0]
    return det / (np.linalg.norm(L) * np.linalg.norm(R))

def find_eigenvalues(omega_min=OMEGA_MIN, omega_max=OMEGA_MAX, nscan=N_SCAN, p=P):
    grid = np.linspace(omega_min, omega_max, nscan)
    vals = np.array([mismatch(w, p) for w in grid])

    roots = []
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            roots.append(grid[i])
        elif vals[i] * vals[i+1] < 0.0:
            root = brentq(lambda w: mismatch(w, p), grid[i], grid[i+1],
                          xtol=1e-11, rtol=1e-12, maxiter=100)
            if not roots or abs(root - roots[-1]) > 1e-7:
                roots.append(root)
    return grid, vals, np.array(roots)

def eigenfunction_profile(omega, nleft=900, nright=900, p=P):
    """Piecewise two-sided mode, matched and normalized at x=1."""
    sm = np.log(X_MATCH)
    sL = np.linspace(np.log(EPS), sm, nleft)
    sR_desc = np.linspace(np.log(XMAX), sm, nright)

    solL = solve_ivp(
        lambda s, y: rhs_logx(s, y, omega, p),
        (sL[0], sL[-1]),
        frobenius_state_zero(EPS, omega, p=p),
        t_eval=sL,
        method="DOP853", rtol=RTOL, atol=ATOL
    )
    solR = solve_ivp(
        lambda s, y: rhs_logx(s, y, omega, p),
        (sR_desc[0], sR_desc[-1]),
        frobenius_state_infinity(XMAX, omega, p=p),
        t_eval=sR_desc,
        method="DOP853", rtol=RTOL, atol=ATOL
    )

    if not solL.success or not solR.success:
        raise RuntimeError("profile integration failed")

    # Reverse right branch so s increases from 0 to log(XMAX).
    sR = solR.t[::-1]
    YR = solR.y[:, ::-1]

    # At an eigenvalue the two match vectors are collinear. Find the scalar
    # that best overlays the right branch on the left branch.
    Lm = solL.y[:, -1]
    Rm = YR[:, 0]
    scale = np.dot(Lm, Rm) / np.dot(Rm, Rm)
    YR *= scale

    s = np.concatenate([solL.t, sR[1:]])
    y = np.concatenate([solL.y[0], YR[0, 1:]])

    # Global amplitude is arbitrary.
    y /= np.max(np.abs(y))

    x = np.exp(s)
    rho = x / (1.0 + x)  # compactifies x in (0,infinity) to rho in (0,1)
    return rho, x, y

# ----------------------- run, save data, make plots ----------------------- #

def main():
    print("Parameters:", P)
    print("Using u(omega) = -omega")
    print("Shooting contour: z=-x, x in (0,infinity)")
    print("Frobenius branches: x^(1/2+a0) at 0 and x^(1/2-ainf) at infinity")

    grid, vals, roots = find_eigenvalues()
    print(f"\nFound {len(roots)} eigenvalues in [{OMEGA_MIN}, {OMEGA_MAX}]:")
    for n, w in enumerate(roots):
        print(f"  n={n:2d}  omega={w:.12f}  D(omega)={mismatch(w):+.3e}")

    # Save eigenvalues.
    eig_csv = OUTDIR / "heun_eigenvalues.csv"
    with eig_csv.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["n", "omega", "u=-omega", "mismatch"])
        for n, w in enumerate(roots):
            wr.writerow([n, f"{w:.15g}", f"{-w:.15g}", f"{mismatch(w):.6e}"])

    # Plot the spectral condition.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(grid, vals, lw=1.2, label=r"$D(\omega)$")
    ax.axhline(0.0, lw=0.8)
    ax.scatter(roots, np.zeros_like(roots), s=28, zorder=3, label="eigenvalues")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"normalized Wronskian $D(\omega)$")
    ax.set_title("Heun shooting eigenvalue condition")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "heun_spectrum.png", dpi=180)
    fig.savefig(OUTDIR / "heun_spectrum.pdf")
    plt.close(fig)

    # Eigenfunctions and sampled data.
    nplot = min(N_MODES_TO_PLOT, len(roots))
    profiles = []
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for n in range(nplot):
        rho, x, y = eigenfunction_profile(roots[n])
        profiles.append((rho, x, y))
        ax.plot(rho, y, lw=1.4, label=fr"$n={n}$, $\omega={roots[n]:.3f}$")

    ax.set_xlabel(r"$\rho=x/(1+x)$  with  $z=-x$")
    ax.set_ylabel(r"normalized $\Psi_n$")
    ax.set_title("First Heun eigenfunctions from two-sided Frobenius shooting")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUTDIR / "heun_eigenfunctions.png", dpi=180)
    fig.savefig(OUTDIR / "heun_eigenfunctions.pdf")
    plt.close(fig)

    # All profile grids are identical because the same s-grids are used.
    prof_csv = OUTDIR / "heun_eigenfunctions.csv"
    with prof_csv.open("w", newline="") as f:
        wr = csv.writer(f)
        header = ["rho", "x"] + [f"psi_n{n}_omega_{roots[n]:.9f}" for n in range(nplot)]
        wr.writerow(header)
        rho0, x0, _ = profiles[0]
        for i in range(len(rho0)):
            wr.writerow([f"{rho0[i]:.15g}", f"{x0[i]:.15g}"] +
                        [f"{profiles[n][2][i]:.15g}" for n in range(nplot)])

    # Also save the scan itself.
    scan_csv = OUTDIR / "heun_mismatch_scan.csv"
    with scan_csv.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["omega", "D(omega)"])
        wr.writerows(zip(grid, vals))

    print("\nWrote:")
    for pth in sorted(OUTDIR.iterdir()):
        print(" ", pth.name)

if __name__ == "__main__":
    main()
