# LAMG+ SISC review (2026-10-31): honest assessment and response plan

Decision: **revision requested** (AE: Ben Southworth; due **2026-12-29**). Referee 2 positive (presentation
only). Referee 1 skeptical: novelty, affinity vs SOC, the RHS `b = L x_true`, the GKS "LAMG fails" rebuttal,
Table 4.3, the scaling-cloud regression. This document records what was *measured* on 2026-10-31 to answer
each major point, what the Brandt corpus says, and the resulting plan. Raw data: `results/revision2026/`.
Everything below was run serially on the M5 Pro with the current `main` (no code changes).

## 0. Bottom line

The work is salvageable and is a real, if modest, contribution — but **the referee is right more often than
not**, and two of his points expose weaknesses in our own methodology that he did not fully see:

| # | Referee-1 point | Verdict | Evidence |
|---|---|---|---|
| 1 | Novelty: straightforward fix for one PDE family | **Half right.** Conceptual novelty is limited (the caliber-1 ceiling is Brandt 1986 Ex. 4.2; the remedy is foreshadowed in LAMG 2012 §6.1; the LS weight is Bootstrap AMG). The contribution is the *execution*: selection rule, two-level quantification, multilevel evidence, and the benchmark. | Brandt-corpus review (§1 below) |
| 2 | Affinity vs matrix SOC; randomness | **Wrong on principle, but his testable worry is answerable with data.** The affinity inversion is a small-K sampling artifact: it vanishes at K=16 *without* the veto. The veto is exactly what lets K stay at 4. Seed-variance is small. | K-sweep §2 |
| 3 | `b = L x_true` favours AMG | **Right in substance.** With random zero-mean b, LAMG+ needs 1.3× more cycles on geomean over the 202 large graphs (approxChol 1.1× more iterations), and its geomean speedup vs approxChol drops from 0.90× to 0.80×; on web-Stanford LAMG+ goes from 5 cycles to 71 (ACF 0.85). Qualitative picture unchanged: parity with approxChol, 1.7–1.9× faster than AC, all 202 converge under both RHSs. Also: our reported "ACF" values are first-cycle transients, not asymptotic rates. GKS use the same `b = M g` protocol as we do, so the *comparison* with GKS is like-for-like, but the referee's point stands. | RHS study §3 |
| 4 | "[13] mis-reported LAMG" needs more than "it ran for us" | **Right that the paper under-argues it; the mechanism is now identified from GKS's public code.** Their harness writes `relres = Inf` *before* the solve and runs MATLAB under `gtimeout`; and LAMG 2.2.1's `errorReductionTol` is an error-norm criterion that, with GKS's random RHS, returns `success=1` at relative residuals up to 3.6e-4 — which GKS's residual check then records as "did not reach tolerance". The original LAMG converges (ACF 0.15–0.48) on every family. | §4 |
| 5 | Table 4.3: LAMG+ never the fastest mean | **Right as stated; the paper over-claims "fastest robust solver on FE".** On large FE, pyAMG matches or beats LAMG+ on 5 of 8 instances; BoomerAMG is 2–6× faster on 3-D grids. The defensible claim is *robust-and-never-catastrophic* (only LAMG+, approxChol, AC), not fastest. | large-instance rerun §5 |
| 6 | Linear scaling from a regression through a cloud | **Right.** Replaced by within-family refinement sequences: cycles and ACF flat in n for every family; per-edge wall time flat within 1.5–2× (cache). One genuine outlier (delaunay_n18, ACF 0.4–0.6 for every seed/RHS). | §6 |

Additional finding the referee did not raise: the paper says caliber-2 is "inert (≈0% of nodes) elsewhere".
**That is false** beyond the finest level of the isotropic 5-point grid: caliber-2 fires on 10–20% of
aggregation-level rows on road networks, Delaunay meshes, FE thermal matrices and even the isotropic grid's
coarse levels, with neutral ACF and 0–25% extra solve time. Must be corrected (§2b).

## 1. Novelty (Brandt corpus)

- Caliber-1 ceiling ρ ≥ 1/2 with no cure by smoothing: Brandt 1986 *Algebraic multigrid theory: the symmetric
  case* §4.7 Ex. 4.2; multilevel degradation 1984 Guide §6.2 eq. (6.1) (χ = 1/2, γ = 1.5 ⇒ γ(1−χ) < 1 ⇒ V/γ-cycle
  factor → 1 with level count — exactly our ACF 0.99). LAMG 2012 §3.4.3 (ρ ≈ 1 − 1/q). Interpolation-order rule:
  Guide §4.3; Brandt 1986 §5.2 (linear *and convex*).
- "Raise the caliber at troublesome nodes": LAMG 2012 §3.1.3 and §6.1 — an explicitly stated open item.
  LS weight from the same test vectors: LAMG 2012 §3.3.4, Bootstrap AMG §2 (K ≥ caliber; K = 4 = 2·caliber satisfies
  the sufficiency rule — say so).
- Positioning for the revision: "we execute the LAMG 2012 §6.1 program for the locally one-dimensional case":
  (i) a selection rule, (ii) a two-level analysis tied to Ex. 4.2, (iii) multilevel evidence, (iv) a large
  same-machine benchmark. Drop "we prove", "here reproduction is discovery", and all headline language.
- Gap Brandt would flag: show a per-level table on the anisotropic grid with/without energy correction and
  recombination, to establish that LAMG's existing tools cannot remove the ceiling (not yet done; needs the
  energy-correction code path, which LAMG+ does not currently implement).
- Rotated anisotropy remains open (LAMG 2012 §5.2 Table 5.3). A 45°-diagonal anisotropic grid: caliber-2 helps
  (ACF 0.37 → 0.20) but does not reach the aligned-case 0.11. State this boundary.

## 2. Affinity vs SOC — K-sweep and seed variance (`ksweep.log`)

256² grid, ε = 1e-4, caliber-2 on, three seeds each:

| K | veto off: ACF | veto on: ACF |
|---|---|---|
| 4 | 0.993 0.992 0.992 (stalled, 60 cycles) | 0.155 0.141 0.138 (7 cycles) |
| 8 | 0.985 0.985 0.985 (stalled) | 0.098 0.097 0.095 (6) |
| 16 | 0.098 0.096 0.097 (6) | identical |
| 32 | 0.104 0.102 0.104 (6) | identical |

So the "affinity inversion" is a small-sample artifact of K = 4 (and 8), gone at K ≥ 16 with no veto at all.
Response: affinity is the primary, principled measure (Guide §1.1 Thm 1.1; LAMG §3.3; Ron–Safro–Brandt Table 3.1
predicts exactly this misordering at small K; Brandt 2000 §3 treats matrix SOC as the *fallback*). The veto is a
0.05-threshold guardrail that buys K = 4 leanness (a 4× cheaper setup than K = 16) at zero cost. Add the
back-of-envelope rate: for a decoupled pair, affinity = 1 − r² with r² ~ Beta(1/2,(K−1)/2) under the null, so
P(affinity > 0.9) ≈ 1e-2 at K = 4 — hundreds of inversions expected on a 256² grid, i.e. predictable, not
"unpredictable".

Seed variance (default config, 8 seeds): aniso 128² ACF 0.095–0.104, cycles 6 always; iso 128² 0.020–0.096
(one seed at 0.096), cycles 5–6; 3-D 32³ 0.170–0.186, cycles 9 always.

### 2b. Caliber-2 is NOT inert off the anisotropic grid (`cal2_fire.log`)

Fraction of fine rows with two parents, over aggregation levels (caliber-2 on vs off; veto on in both):

| graph | 2-parent rows | cycles / ACF with | cycles / ACF without | solve s with / without |
|---|---|---|---|---|
| aniso 256² (control) | 29% | 7 / 0.138 | 13 / 0.523 | 0.12 / 0.19 |
| iso 256² (control) | 9.4% (0.5% finest, ~20% coarser) | 5 / 0.015 | 5 / 0.016 | 0.16 / 0.16 |
| diag-aniso 256² (45°) | 29% | 10 / 0.200 | 15 / 0.365 | 0.21 / 0.43 |
| luxembourg_osm | 13% | 4 / 0.007 | 4 / 0.006 | 0.02 / 0.02 |
| roadNet-PA | 19.5% | 5 / 0.015 | 5 / 0.021 | 2.38 / 1.91 |
| delaunay_n17 | 9.7% | 5 / 0.018 | 5 / 0.019 | 0.77 / 0.69 |
| fe_ocean | 3.6% | 5 / 0.015 | 5 / 0.015 | 0.94 / 0.82 |
| thermal1 | 9.1% | 5 / 0.020 | 5 / 0.021 | 0.55 / 0.49 |

The paper's "≈0% of nodes upgrade on isotropic graphs" measured only the finest level of the 5-point grid.
Revision: report the per-level firing fraction honestly; the gate is *harmless* (ACF unchanged) rather than
*inert*, and costs ≤ 25% solve time on roads. Consider restricting the upgrade to nodes whose two parents are
reached through edges of *comparable* strength ratio to the vetoed ones (true 1-D), which would make it inert
on isotropic meshes; that is a code change and must be re-benchmarked.

## 3. Right-hand side (`rhs_study.csv`, `true_acf2.log`)

GKS's protocol (their §5): "apply M to a random Gaussian g and normalize" — the same `b = L x` family we use.
So Table 4.7 and the GKS comparison are like-for-like. But the referee's mechanism is real: with `b = L x`,
`x` random, the error is white in x-space, the residual norm is dominated by high-frequency error, and the
relative-residual criterion barely tests smooth-error reduction. Three RHS types × 3 seeds × 14 systems:

| RHS | LAMG+ cycles (geomean ratio to LX) | approxChol its ratio | AC its ratio | LAMG+ solve-time ratio | approxChol solve-time ratio |
|---|---|---|---|---|---|
| RAND (zero-mean Gaussian) | 1.69× | 1.22× | 1.21× | 1.77× | 1.12× |
| DIP (s–t dipole) | 1.39× | 1.14× | 1.13× | 1.40× | 1.20× |

Per-system (LX → RAND): web-Stanford 5.7 → 35 cycles (ACF 0.13 → 0.59); wted_chimera 4 → 9; grid2_aniso
6 → 17; delaunay_n18 9 → 22; FE (bmwcra_1, pwtk) 4–5 → 6; com-DBLP 4 → 5; roadNet-PA 4.7 → 7.

Properly measured asymptotic factors (run to 1e-12, factor over the 1e-4…1e-11 window):

| graph | LX: cycles to 1e-8 / ACF | RAND: cycles to 1e-8 / ACF | approxChol its LX / RAND |
|---|---|---|---|
| grid2d 512² | 5 / 0.049 | 7 / 0.061 | 39 / 44 |
| aniso 256² | 7 / 0.166 | 9 / (hits 1e-10 floor, then escalation-restart loop) | 16 / 26 |
| grid3d 64³ | 9 / 0.174 | 10 / 0.176 | 26 / 27 |
| chimera 1e5 | 5 / 0.045 | 6 / 0.051 | 37 / 40 |
| delaunay_n16 | 5 / 0.053 | 6 / 0.065 | 29 / 32 |
| **delaunay_n18** | 8 / **0.480** | **27 / 0.608** | 33 / 37 |
| delaunay_n20 | 5 / 0.083 | 8 / 0.148 | 35 / 40 |
| roadNet-PA | 5 / 0.066 | 6 / 0.080 | 38 / 45 |
| com-DBLP | 4 / 0.027 | 5 / 0.034 | 22 / 24 |
| **web-Stanford** | 5 / **0.273** | **not in 60 cycles / 0.845** | 47 / 62 |
| bmwcra_1 | 4 / 0.029 | 5 / 0.038 | 13 / 14 |
| cage12 | 7 / 0.093 | 8 / 0.095 | 17 / 18 |
| thermal2 | 5 / 0.066 | 7 / 0.086 | 38 / 44 |

Conclusions: (a) for most classes the RHS effect is one or two cycles and the ranking is unchanged;
(b) the paper's "ACF" column (geomean of the last 5 factors of a 5-cycle solve) is **not asymptotic** —
e.g. web-Stanford's reported 0.127 is 0.27 asymptotically, delaunay-type meshes reported ~0.02 can be 0.5;
(c) two genuine slow cases exist that `b = L x` hides: web crawls (web-Stanford fails to reach 1e-8 in 60
cycles with random b) and one Delaunay refinement level; (d) the γ-escalation "restart from initial guess"
heuristic produces residual blow-ups (×10⁹) when triggered by a roundoff floor and does not help web-Stanford.

Plan: adopt the harder protocol — zero-mean random `b` (LAMG 2012 §5.1 protocol too) — for *all* tables, report
both cycles-to-1e-8 and a properly measured asymptotic factor (1e-4…1e-11 window), keep `b = L x` only for the
GKS-family table (their protocol), and add a "slow tail" paragraph naming web crawls honestly. Decide whether to
disable the escalation restart or gate it on the residual being far above the floor.

## 4. GKS [13] and the original LAMG (`octave_rand/`, `octave_rand_tight/`)

What GKS's public code does (`Laplacians.jl/compare/matlabSafe.jl`, `matlab/timeLamg.m`):
1. `timeLamg.m` begins with `st = bt = iter = relres = Inf; save toJulia ...` — the failure values are written
   *before* the solve.
2. MATLAB is run as `gtimeout $(limit) matlab -nojvm < timeLamg.m`; if killed, `toJulia.mat` still holds the
   Inf defaults, so a timeout is recorded identically to non-convergence. The limit is not stated in the paper.
3. The solver call is `lamg.solve(setup, b, 'errorReductionTol', tol)` with tol = 1e-8 and `relres =
   norm(la*x-b)/norm(b)` evaluated afterwards; GKS then report "unable to converge to our target tolerance"
   (footnote 1) with no row for LAMG.
4. LAMG 2.2.1's `errorReductionTol` stops when the (eliminated-level) residual norm is reduced by 1e-8 relative
   to its own initial value, with `success = 1`. With GKS's random RHS this lands the *finest-level* relative
   residual well above 1e-8 on half the families.

Original LAMG 2.2.1 (Octave, algorithm/MEX unchanged), GKS call, **GKS-style random zero-mean b**:

| family | cycles | relres at exit | success | ACF | relres with tol 1e-11 |
|---|---|---|---|---|---|
| chimera | 11 | 2.9e-9 | 1 | 0.155 | 1.1e-11 |
| uni chimera | 12 | 4.7e-9 | 1 | 0.178 | 4.9e-12 |
| wted chimera | 11 | 1.8e-8 | 1 | 0.172 | 1.5e-11 |
| SDDM chimera | 11 | 7.9e-9 | 1 | 0.169 | 1.3e-9 |
| star | 4 | 4.6e-8 | 1 | 0.002 | 5.5e-11 |
| grid 2D uniform | 14 | 2.6e-9 | 1 | 0.239 | 3.5e-11 |
| grid 2D aniso | 21 | 2.2e-7 | 1 | 0.478 | 2.2e-7 (hits cap) |
| grid 2D hicon | 12 | 9.4e-8 | 1 | 0.223 | 1.5e-10 |
| grid 3D uniform | 5 | **3.6e-4** | 1 | 0.176 | 3.6e-4 |
| grid 3D aniso | 20 | 6.6e-9 | 1 | 0.386 | 2.6e-9 |
| grid 3D hicon | 12 | 1.4e-6 | 1 | 0.285 | 5.1e-9 |
| SPE10 | 11 | 4.2e-5 | 1 | 0.202 | 1.6e-7 |

So: the algorithm converges on every family (ACF 0.002–0.48), and a harness that only checks the final
relative residual against 1e-8 will nonetheless classify 5–6 of 12 as "did not reach tolerance" because of
the *stopping-criterion mismatch* — plus whatever the timeout killed. This is the precise, verifiable
statement to make. The user's instinct is right: the point is reproducibility, and both the mechanism and the
data are now scripted. Do **not** say "bug"; say "evaluation protocol: a residual target applied to a solver
whose termination test is an error-reduction target, under a kill timeout that records as non-convergence".
Also note the 3-D uniform case (relres 3.6e-4 after 5 cycles, tol 1e-11 does not help): the eliminated-level
criterion can terminate far from the fine residual target — a genuine usability defect of LAMG 2.2.1's API
(not of the multigrid), which LAMG+ fixes by testing the finest-level residual.

## 5. Table 4.3 / "never the fastest" — large-instance rerun with BoomerAMG and pyAMG (`large_compete.csv`)

µs per nonzero, setup+solve to 1e-8, serial, warm, all converged (bold = fastest):

| instance (nnz) | class | LAMG+ | approxChol | AC | BoomerAMG | pyAMG |
|---|---|---|---|---|---|---|
| bmwcra_1 (1.1e7) | FE | 0.104 | 0.359 | 0.820 | 0.239 | **0.095** |
| crankseg_1 (1.1e7) | FE | **0.081** | 0.361 | 1.04 | 0.602 | 0.128 |
| troll (1.2e7) | FE | **0.102** | 0.332 | 0.863 | 0.191 | 0.160 |
| pwtk (1.2e7) | FE | 0.104 | 0.224 | 0.612 | 0.142 | **0.094** |
| af_shell8 (1.8e7) | FE | 0.176 | 0.241 | 0.467 | 0.123 | **0.089** |
| inline_1 (3.7e7) | FE | **0.101** | 0.295 | 0.728 | 0.339 | 0.194 |
| Fault_639 (2.7e7) | FE | 0.100 | 0.431 | 1.29 | 0.217 | **0.098** |
| bone010 (4.8e7) | FE | 0.187 | 0.546 | 1.43 | 0.350 | **0.092** |
| apache2 (4.8e6) | mesh | 1.16 | 0.361 | 0.928 | **0.213** | 0.482 |
| G3_circuit (7.7e6) | mesh | 0.973 | 0.336 | 0.603 | **0.296** | 1.11 |
| thermal2 (8.6e6) | mesh | 1.34 | 0.364 | 0.585 | **0.264** | 0.657 |
| parabolic_fem (3.7e6) | mesh | 1.76 | 0.374 | 0.600 | **0.183** | 0.366 |
| ecology1 (5.0e6) | mesh | 0.824 | 0.307 | 0.592 | **0.211** | 0.365 |
| delaunay_n20 (7.3e6) | mesh | 1.25 | 0.337 | 0.608 | **0.268** | 1.19 |
| roadNet-CA (7.5e6) | road | 0.762 | **0.338** | 0.489 | 0.395 | 1.96 |
| com-DBLP (2.4e6) | social | 0.464 | **0.287** | 0.532 | 1.56 | 26.7 |
| web-Google (5.9e6) | web | 0.629 | **0.394** | 0.681 | 1.20 | 24.2 |

Honest reading: on large FE/structural matrices LAMG+ is 2–4× faster than approxChol and 6–10× faster than
AC (the paper's claim against the near-linear solvers holds), but pyAMG is as fast or faster on 5 of 8, and
on the 2-D/3-D PDE-mesh class (apache2, thermal2, parabolic_fem, delaunay) LAMG+ is the **slowest of the
five**: 3–7× behind BoomerAMG and ~3× behind approxChol. On social/web, pyAMG does converge here but at
24–27 µs/nnz (40–60× slower), BoomerAMG at 2–3×; approxChol is fastest. The claim "fastest robust solver" is only true under the
definition "robust = converges on every class", which pyAMG (DNF on social/web) and BoomerAMG (10× off-turf)
fail. Revision: (i) say exactly that, with the DNF counts; (ii) quantify the referee's portfolio: an
oracle "best of {BoomerAMG, pyAMG, AC}" vs LAMG+ vs a realistic degree-classifier portfolio (LAMG 2012 §5.1
anticipated this objection; Brandt's robustness-first stance, Guide §13.1, is the right frame, but it needs
numbers, not philosophy).

## 6. Scaling (`scaling_family2.csv`)

Within-family refinement sequences, LAMG+ cycles / ACF / total µs per edge, approxChol µs per edge:

| family | sizes (m) | cycles | ACF | LAMG+ µs/edge | approxChol µs/edge |
|---|---|---|---|---|---|
| 2-D grid | 3e4 → 8.4e6 | 5,5,5,5,5 | 0.015–0.023 | 1.07 → 2.15 | 0.55 → 0.91 |
| 2-D aniso ε=1e-4 | 3e4 → 8.4e6 | 6,7,7,7,6 | 0.09–0.14 | 1.05 → 2.29 | 0.22 → 0.64 |
| 3-D grid | 1e4 → 6.2e6 | 5,9,9,9 | 0.016–0.18 | 1.39 → 3.43 | 0.39 → 1.12 |
| chimera | 2e4 → 1.4e7 | 4–7 | 0.006–0.33 | 0.7–2.3 | 0.46–1.9 |
| delaunay n10–n20 | 3e3 → 3.1e6 | 5 except **n18: 8–12** | 0.015–0.02 except **n18: 0.4–0.6** | 1.2 → 3.2 | 0.29 → 0.87 |
| rgg 15–19 | 1.6e5 → 3.3e6 | 4,4,4,7,7 | 0.005–0.11 | 0.8–1.6 | 0.46–0.70 |
| roads PA/TX/CA/GB/DE | 1.5e6 → 1.2e7 | 5,5,5,4,5 | 0.009–0.022 | 2.4, 2.3, 2.4, 0.91, 1.14 | 1.04, 1.19, 1.17, 1.34, 1.64 |

Cycle count and convergence factor are size-independent within every family (the delaunay_n18 level is a
hierarchy-quality outlier for every seed and RHS; not the level cap — verified with max_levels = 40).
Per-edge wall time grows ~1.5–2.5× from cache-resident to DRAM-resident sizes, for both solvers. On these
structured families approxChol is 2–5× faster; LAMG+ wins only on the country-scale OSM roads (1.4×) and the
dense 1.4e7-edge chimera (2.8×). Revision: replace Fig. 4.1's regression line with these tables/plots, keep
the cloud as a boundedness figure (max/percentiles, no fit), and report operator complexity per family.

## 7. The 201-graph head-to-head under both RHS (`compete_rhs.csv`)

Final, 202 instances with m > 1e6 (nnz ≤ 6e7), serial, warm, tol 1e-8, CSV `results/revision2026/compete_rhs.csv`.

| RHS | LAMG+ vs approxChol (geomean, win %) | LAMG+ vs AC (geomean, win %) | LAMG+ cycles median / max | approxChol its median / max | converged |
|---|---|---|---|---|---|
| LX (b = L x, paper + GKS protocol) | 0.90× / 48% | 1.89× / 75% | 5 / 17 | 17 / 62 | 202/202 |
| RAND (zero-mean Gaussian) | 0.80× / 45% | 1.67× / 73% | 6 / 71 | 17 / 84 | 202/202 |

RAND/LX cycle ratio: LAMG+ geomean 1.32, approxChol iteration ratio 1.09. So the referee is right in direction:
the LX right-hand side flatters LAMG+ by ~10 percentage points of geomean speedup, and the switch costs LAMG+ about
one extra cycle per solve on typical graphs. It does not change the qualitative picture: LAMG+ is roughly at parity
with approxChol (slightly slower on geomean, wins just under half), and clearly faster than AC, under both RHSs.
Every instance converges to 1e-8 under both RHSs (max_cycles 300).

Worst LAMG+ cases under RAND (cycles LX → RAND, ACF): web-Stanford 5 → 71 (0.85), Kamvar/Stanford 17 → 38 (0.67),
cnr-2000 9 → 34 (0.59), NLR 9 → 30 (0.82), largebasis 5 → 19 (0.29), web-BerkStan 5 → 16 (0.46). The web graphs are
the one family where LAMG+'s asymptotic factor is genuinely poor (0.6–0.85) and the LX protocol hid it; this must be
said in the paper, with the RAND numbers reported alongside LX.

## 8. Revision plan (priority order)

1. Re-run every table with zero-mean random `b` and report (cycles to 1e-8, asymptotic factor measured in the
   1e-4…1e-11 window, time/edge); keep `b = L x` only where reproducing GKS. Add the slow-tail paragraph
   (web crawls; delaunay_n18-type outliers) and fix or gate the escalation restart.
2. Replace §4.2 with within-family refinement sequences + operator complexity; cloud becomes a robustness figure.
3. §4.5/§4.3: re-state the FE claim as "fastest of the robust near-linear solvers"; add the large-instance
   five-solver table; add an oracle-portfolio vs realistic-portfolio comparison.
4. §4.7 (GKS): rewrite as a reproducibility note: the harness mechanism (Inf-before-solve, gtimeout), the
   criterion mismatch with the random-RHS table above, all scripted. No speculation about bugs.
5. §3: reposition novelty per §1 (Brandt 1986 Ex. 4.2, Guide §6.2 (6.1), LAMG 2012 §6.1, Bootstrap AMG §2);
   add the K-sweep table and the Beta-tail rate; correct "inert ≈0%" to the per-level firing table; add the
   45° boundary case; state rotated anisotropy as open.
6. Referee 2 and all style points: full sweep (user).
7. Response letter: concede 1 (partly), 3, 4 (wording), 5 (wording), 6; push back on 2 and the "one family"
   half of 1 and 5(a) with the data above.

## 9. Scripts (scratch; copy into `scripts/revision2026/` when adopted)

`rhs_study.jl`, `true_acf2.jl`, `scaling_family2.jl`, `large_compete.jl`, `compete_rhs.jl`, `ksweep.jl`,
`cal2_fire.jl`, `coarsest.jl`, `gks_orig_randb.m`, `gks_orig_randb_tight.m` — all in
`results/revision2026/scripts/`.
