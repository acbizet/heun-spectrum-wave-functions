
"""
Test of the O(1/t) wave-function reconstruction of arXiv:2608.00363,
Sec. 3.5, for the generic Heun problem at t=10000.

Important: unlike the earlier toy run u=-omega, this test uses the
expansion consistent with our paper
    u = u0(kappa) + u1(kappa)/t
from eq. (104), because the closed polynomial f1^{out/in} in eqs.
(120)-(124) is derived together with that expansion.

We shoot the full Heun equation with Frobenius conditions at z=0 and
z=infinity on the negative-real ray z=-x, and select the second
low-lying mode (n=1, when ordered from kappa^2 closest to zero).

We compare:
    full Heun mode
    leading outer/inner hypergeometric patches
    O(1/t)-reconstructed outer/inner patches

Formulas from our paper:
  outer f1: eqs. (120)-(121)
  inner f1: eqs. (123)-(124)
  wave-function reconstruction: eq. (141), with the analogous inner formula
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass(frozen=True)
class Params:
    a0: float = 1.0
    a1: float = 0.2
    at: float = 0.3
    ainf: float = 1.0
    t: float = 10000.0

P = Params()
EPS = 1e-7
XMAX_FACTOR = 100.0
XMAX = P.t * XMAX_FACTOR
XM = np.sqrt(P.t)
RTOL_SCAN = 4e-8
RTOL = 1e-10
ATOL = 1e-11
TARGET_MODE = 1

OUTDIR = Path(__file__).resolve().parent / "output"
OUTDIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Paper-consistent accessory parameter, eq. (104)
# ------------------------------------------------------------

def u0(k2, p=P):
    return -0.25 - p.ainf**2 + p.at**2 + k2

def u1(k2, p=P):
    numL = -1.0 - 4*p.a0**2 + 4*p.a1**2 + 4*k2
    numR = -1.0 - 4*p.ainf**2 + 4*p.at**2 + 4*k2
    return numL * numR / (32*k2 - 8)

def u_full(k2, p=P):
    return u0(k2,p) + u1(k2,p)/p.t


# ------------------------------------------------------------
# Full and leading hypergeometric potentials
# ------------------------------------------------------------

def Q_full(z, k2, p=P):
    u = u_full(k2,p)
    C = 0.5 - p.a0**2 - p.a1**2 - p.at**2 + p.ainf**2 + u
    return (
        (0.25-p.a0**2)/z**2
        + (0.25-p.a1**2)/(z-1)**2
        + (0.25-p.at**2)/(z-p.t)**2
        - C/(z*(z-1))
        + u/(z*(z-p.t))
    )

def lambda_out0(k2, p=P):
    # Leading t=infinity value; use u0, not u_full.
    return -p.at**2 + p.ainf**2 - p.a0**2 - p.a1**2 + u0(k2,p) + 0.5

def Q_outer0(z, k2, p=P):
    lam = lambda_out0(k2,p)
    return (
        (0.25-p.a0**2)/z**2
        + (0.25-p.a1**2)/(z-1)**2
        - lam/(z*(z-1))
    )

def Q_inner0(wp, k2, p=P):
    # Leading inner potential, with u0 and lambda_in=-u0.
    uu = u0(k2,p)
    lam = -uu
    return (
        -lam/(wp*(wp-1))
        + (0.25-p.at**2)/(wp-1)**2
        + (p.at**2-p.ainf**2-uu)/wp**2
    )

def q_inner0_x(x, k2, p=P):
    wp = -x/p.t
    return Q_inner0(wp,k2,p)/p.t**2


# ------------------------------------------------------------
# Integrator in s=log x, z=-x
# ------------------------------------------------------------

def rhs(s, Y, k2, model, p=P):
    x = np.exp(s)
    if model == "full":
        q = Q_full(-x,k2,p)
    elif model == "outer":
        q = Q_outer0(-x,k2,p)
    elif model == "inner":
        q = q_inner0_x(x,k2,p)
    else:
        raise ValueError(model)
    return np.array([Y[1], Y[1]-x*x*q*Y[0]])

def integrate(k2, model, side, rtol=RTOL, dense_output=False, p=P):
    sm = np.log(XM)
    if side == "left":
        span = (np.log(EPS),sm)
        Y0 = np.array([1.0,0.5+p.a0])
    else:
        span = (np.log(XMAX),sm)
        Y0 = np.array([1.0,0.5-p.ainf])
    sol = solve_ivp(
        lambda s,y: rhs(s,y,k2,model,p),
        span,Y0,method="DOP853",
        rtol=rtol,atol=ATOL,dense_output=dense_output
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol

def mismatch(k2):
    L = integrate(k2,"full","left",rtol=RTOL_SCAN).y[:,-1]
    R = integrate(k2,"full","right",rtol=RTOL_SCAN).y[:,-1]
    W = L[0]*R[1]-L[1]*R[0]
    return W/(np.linalg.norm(L)*np.linalg.norm(R))


# ------------------------------------------------------------
# Find low modes.  We scan kappa^2<0 and order roots from zero outward.
# ------------------------------------------------------------

def spectrum():
    g = np.linspace(-7.0,0.15,220)
    D = np.array([mismatch(q) for q in g])
    roots=[]
    for a,b,fa,fb in zip(g[:-1],g[1:],D[:-1],D[1:]):
        if fa*fb < 0:
            r=brentq(mismatch,a,b,xtol=1e-11,rtol=1e-11)
            if not roots or abs(r-roots[-1])>1e-7:
                roots.append(r)
    # low-mode ordering: closest to kappa^2=0 first
    roots=np.array(sorted(roots,reverse=True))
    return g,D,roots


# ------------------------------------------------------------
# f1 polynomials from eqs. (120)-(124)
# ------------------------------------------------------------

def outer_f1_coeffs(k2,p=P):
    den = 2-8*k2
    A12=(-1-4*p.ainf**2+4*p.at**2+4*k2)/den
    A11=( 1+4*p.ainf**2-4*p.at**2-4*k2)/den
    return A12,A11

def inner_f1_coeffs(k2,p=P):
    den = 2-8*k2
    B11=(-1-4*p.a0**2+4*p.a1**2+4*k2)/den
    B10=( 1+4*p.a0**2-4*p.a1**2-4*k2)/den
    return B11,B10


# ------------------------------------------------------------
# Build full, leading, and corrected profiles
# ------------------------------------------------------------

def profiles(k2,x,p=P):
    FL=integrate(k2,"full","left",dense_output=True)
    FR=integrate(k2,"full","right",dense_output=True)
    OL=integrate(k2,"outer","left",dense_output=True)
    IR=integrate(k2,"inner","right",dense_output=True)

    # Global right normalization from the exact eigenfunction.
    Lm=FL.y[:,-1]
    Rm=FR.y[:,-1]
    right_scale=np.dot(Lm,Rm)/np.dot(Rm,Rm)

    s=np.log(x)
    left=x<=XM

    Yfull=np.empty_like(x)
    Yfull[left]=FL.sol(s[left])[0]
    Yfull[~left]=right_scale*FR.sol(s[~left])[0]

    O=OL.sol(s)
    I=IR.sol(s)
    yout0=O[0]
    yin0=right_scale*I[0]

    # Derivatives stored by the s-equation: O[1] = d psi / d log x.
    # z=-x, so d/dz = -(1/x)d/d(log x).
    z=-x
    A12,A11=outer_f1_coeffs(k2,p)
    fout=A12*z*z+A11*z
    fpout=2*A12*z+A11
    dpsi_dz=-O[1]/x

    # Eq. (141):
    # Psi = psi - (1/t)( f1 psi' - 1/2 f1' psi )
    yout1=yout0-(fout*dpsi_dz-0.5*fpout*yout0)/p.t

    # Inner analogue in wp=z/t=-x/t.
    wp=-x/p.t
    B11,B10=inner_f1_coeffs(k2,p)
    fin=B11*wp+B10
    fpin=B11*np.ones_like(wp)

    # d psi / d wp = -t/x * d psi/dlog x.
    dpsi_dwp=-(p.t/x)*I[1]
    yin_raw1=I[0]-(fin*dpsi_dwp-0.5*fpin*I[0])/p.t
    yin1=right_scale*yin_raw1

    # Each patch has arbitrary overall normalization. Correct the tiny
    # O(1/t) normalization shift induced by the inverse diffeomorphism,
    # using a reference point deep inside the patch.
    #
    # This tests shape improvement, not amplitude convention.
    iout=np.argmin(abs(x-1e-4))
    cout=yout0[iout]/yout1[iout]
    yout1*=cout

    iin=np.argmin(abs(x-1e5))
    cin=yin0[iin]/yin1[iin]
    yin1*=cin

    # Common normalization from the exact mode.
    norm=np.max(np.abs(Yfull))
    return Yfull/norm,yout0/norm,yout1/norm,yin0/norm,yin1/norm


def main():
    grid,D,roots=spectrum()

    print("t =",P.t)
    print("match x=sqrt(t) =",XM)
    print("\nLow modes, ordered from kappa^2 closest to zero:")
    for n,k2 in enumerate(roots):
        print(
            f"n={n:2d}  kappa^2={k2:.12f}  "
            f"u0={u0(k2):.12f}  u1/t={u1(k2)/P.t:+.5e}  "
            f"D={mismatch(k2):+.2e}"
        )

    k2=roots[TARGET_MODE]
    print("\nSelected mode n =",TARGET_MODE)
    print("kappa^2 =",k2)
    print("u(full) =",u_full(k2))

    A12,A11=outer_f1_coeffs(k2)
    B11,B10=inner_f1_coeffs(k2)
    print("A12,A11 =",A12,A11)
    print("B11,B10 =",B11,B10)

    x=np.logspace(-5,6,2600)
    full,o0,o1,i0,i1=profiles(k2,x)

    # Pointwise absolute errors.
    eo0=np.abs(o0-full)
    eo1=np.abs(o1-full)
    ei0=np.abs(i0-full)
    ei1=np.abs(i1-full)

    # Save all data.
    with (OUTDIR/"first_correction_profiles.csv").open("w",newline="") as f:
        wr=csv.writer(f)
        wr.writerow([
            "x_abs_z","Psi_full",
            "outer_leading","outer_O1","outer_err_leading","outer_err_O1",
            "inner_leading","inner_O1","inner_err_leading","inner_err_O1"
        ])
        wr.writerows(zip(x,full,o0,o1,eo0,eo1,i0,i1,ei0,ei1))

    with (OUTDIR/"eigenvalues_paper_consistent.csv").open("w",newline="") as f:
        wr=csv.writer(f)
        wr.writerow(["n","kappa_squared","u0","u1_over_t","u_full","mismatch"])
        for n,q in enumerate(roots):
            wr.writerow([n,q,u0(q),u1(q)/P.t,u_full(q),mismatch(q)])

    # Plot wave functions.
    lo,hi=10.0,1000.0
    fig,ax=plt.subplots(figsize=(9.0,5.6))
    ax.semilogx(x,full,lw=2.2,label="full Heun")
    mo=x<=P.t
    mi=x>=1
    ax.semilogx(x[mo],o0[mo],lw=1.2,ls="--",label="outer leading")
    ax.semilogx(x[mo],o1[mo],lw=1.4,label=r"outer + $O(1/t)$ reconstruction")
    ax.semilogx(x[mi],i0[mi],lw=1.2,ls="--",label="inner leading")
    ax.semilogx(x[mi],i1[mi],lw=1.4,label=r"inner + $O(1/t)$ reconstruction")
    ax.axvspan(lo,hi,alpha=.10,label=r"overlap guide $10<|z|<1000$")
    ax.axvline(XM,lw=.8)
    ax.set_xlabel(r"$x=|z|$ on $z=-x$")
    ax.set_ylabel("normalized wavefunction")
    ax.set_title(
        fr"Paper-consistent first wave-function correction, $t={P.t:g}$, "
        fr"$n={TARGET_MODE}$, $\kappa^2={k2:.6f}$"
    )
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR/"mode_matching_with_first_correction.png",dpi=190)
    fig.savefig(OUTDIR/"mode_matching_with_first_correction.pdf")
    plt.close(fig)

    # Error plot: use ranges where each patch is intended.
    fig,ax=plt.subplots(figsize=(9.0,5.3))
    mask_o=(x>=1e-2)&(x<=3000)
    mask_i=(x>=3)&(x<=1e6)
    floor=1e-14
    ax.loglog(x[mask_o],np.maximum(eo0[mask_o],floor),ls="--",label="outer leading error")
    ax.loglog(x[mask_o],np.maximum(eo1[mask_o],floor),label=r"outer $O(1/t)$ error")
    ax.loglog(x[mask_i],np.maximum(ei0[mask_i],floor),ls="--",label="inner leading error")
    ax.loglog(x[mask_i],np.maximum(ei1[mask_i],floor),label=r"inner $O(1/t)$ error")
    ax.axvspan(lo,hi,alpha=.10)
    ax.axvline(XM,lw=.8)
    ax.set_xlabel(r"$x=|z|$")
    ax.set_ylabel(r"$|\Psi_{\rm approx}-\Psi_{\rm Heun}|$")
    ax.set_title("Does the first diffeomorphism correction improve the wavefunction?")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUTDIR/"first_correction_error.png",dpi=190)
    fig.savefig(OUTDIR/"first_correction_error.pdf")
    plt.close(fig)

    # Quantitative table at representative points.
    pts=[1,3,10,30,100,300,1000,3000,10000]
    with (OUTDIR/"error_at_points.csv").open("w",newline="") as f:
        wr=csv.writer(f)
        wr.writerow([
            "x","outer_leading_error","outer_corrected_error","outer_improvement_factor",
            "inner_leading_error","inner_corrected_error","inner_improvement_factor"
        ])
        for xx in pts:
            j=np.argmin(abs(x-xx))
            fo=eo0[j]/eo1[j] if eo1[j]>0 else np.inf
            fi=ei0[j]/ei1[j] if ei1[j]>0 else np.inf
            wr.writerow([xx,eo0[j],eo1[j],fo,ei0[j],ei1[j],fi])
            print(
                f"x={xx:6g}: "
                f"outer {eo0[j]:.3e}->{eo1[j]:.3e} (x{fo:.2f}); "
                f"inner {ei0[j]:.3e}->{ei1[j]:.3e} (x{fi:.2f})"
            )

    print("\nWrote:")
    for p in sorted(OUTDIR.iterdir()):
        print(" ",p.name)

if __name__=="__main__":
    main()
