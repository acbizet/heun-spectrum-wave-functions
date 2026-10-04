# Astrophysical potential and next physical implementation

## Why this large-`t` machinery is relevant

In our paper, we apply the inner/outer hypergeometric projection to linearized fluctuations of Kerr-de Sitter and Kerr-anti-de Sitter black holes.  In the near-extremal/low-temperature degeneration, the radial Heun problem develops a long throat and splits into a near-horizon (inner) and exterior (outer) description.  The information missing from either local description is encoded in Robin data in the overlap region.

This creates a potentially powerful numerical/analytic hybrid strategy: solve a simple hypergeometric problem in each patch, incorporate finite-temperature corrections through the diffeomorphism and matching data, and use a direct Heun solver only as an independent benchmark.

## Astrophysical/black-hole observables that could benefit

### 1. Near-extremal quasinormal spectra

A physical implementation would promote the Heun parameters to their Kerr-(A)dS functions of complex frequency `ω`, azimuthal number `m`, spin weight `s`, horizon data, and the angular separation constant.  The relevant spectrum is then obtained from physical horizon/asymptotic boundary conditions rather than the real toy spectral slices used here.

High-precision numerical shooting would be valuable for:

- validating our all-orders Robin quantization condition;
- determining how rapidly the large-`t` expansion converges away from exact extremality;
- resolving small splittings among low-frequency near-horizon modes;
- benchmarking analytic quasinormal-mode formulae against the full Teukolsky/Heun problem.

### 2. Schwarzian / Jackiw-Teitelboim low modes

Section 4 identifies a subset of Kerr-(A)dS fluctuations that continuously reduce, in the near-extremal limit, to the Schwarzian/JT functional space.  The numerical machinery here can test the finite-temperature continuation of those modes at the level of both eigenvalues and complete radial profiles.

A particularly sharp test would compare the corrected inner solution with a full Kerr radial mode through the overlap and quantify whether the predicted Robin data reproduce the exterior tail without independent fitting.

### 3. Green functions, response and scattering

In our paper, we note that the corrected connection problem can be used to obtain retarded Green functions in the physical problem.  Once the physical parameter map is included, the same numerical framework could benchmark poles, residues and connection coefficients, which are closely related to quasinormal modes, scattering amplitudes and greybody data.

### 4. Superradiant/near-horizon instabilities

In our paper, we explicitly identify the search for other low-temperature Kerr modes, including possible superradiant instabilities, as a future direction.  A two-patch solver with complex `ω` is well suited to tracking such modes as extremality is approached, because the large hierarchy of radial scales is handled analytically rather than by brute-force integration over a stiff domain.

### 5. Post-Minkowskian and post-Newtonian gravitational-wave expansions

The introduction points out that the same interaction-disconnecting strategy may be useful for post-Minkowskian corrections to gravitational interactions and post-Newtonian corrections to gravitational waveforms, potentially avoiding integral/Born-series calculations in some formulations.  The references cited there include work on Kerr-Compton amplitudes and resummed PM/PN waveform expansions.

This repository does **not** yet compute an astrophysical waveform.  Its relevance is methodological: the high-precision test shows that the local hypergeometric wave functions plus the first geometric correction can reproduce a full Heun mode extremely accurately in their overlap regime.

## What must be added before making astrophysical claims

The current high-precision test is a generic Heun calculation.  A physical Kerr-(A)dS implementation should add:

1. the explicit Kerr-(A)dS parameter map of Sec. 4 and Appendix D;
2. complex-frequency root finding for `ω`;
3. the physical radial Frobenius choices (ingoing/outgoing/normalizable, as appropriate);
4. the angular Teukolsky eigenvalue and its coupling to the radial accessory parameter;
5. simultaneous control of radial and angular spectral conditions where necessary;
6. convergence tests in `1/t` and comparison with direct full-Heun/Teukolsky integration;
7. for observables, consistent normalization and extraction of connection coefficients/residues.

Until these steps are implemented, the numerical values in this repository should be read as a precision validation of the **mathematical matching mechanism**, not as Kerr quasinormal frequencies or gravitational-wave predictions.

## Suggested physical benchmark sequence

A practical next program is:

- choose one near-extremal Kerr-dS background where the four regular singular points remain explicit;
- implement the exact `a_i(ω)` and accessory parameter from our paper;
- solve the full complex Heun radial connection problem numerically;
- solve the leading and corrected two-patch problem independently;
- compare complex mode frequencies and radial profiles versus temperature;
- repeat for several multipoles `(s,l,m)`;
- finally compare connection coefficients/Green-function data, not only eigenvalues.

That would turn the present proof-of-principle into a quantitatively astrophysical test.
