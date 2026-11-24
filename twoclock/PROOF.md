# Proof of Proposition 1 (price band, saturation, shedding tiers)

Notation. At step t the state s_t fixes the feasible set Ω_t(s_t) of the fast-clock LP
(wear-derated limits, ramp window around the previous output, battery state-of-charge window,
loads, reserve requirement, network). The LP chooses x (dispatch, flows, battery, imports, shedding),
with each thermal unit split p_g = q_g + e_g, free block q_g ≤ σ r̄_g and excess
e_g ∈ [e̲_g, ē_g] (e̲_g is the ramp-forced minimum excess). The priced objective is

    f(x) + μ · a(x),     a(x) = Σ_g α_g e_g,     α_g = k_g / ((1-σ) r̄_g) ≥ 0,

where f is the fuel + import + shedding cost with nameplate c_g (heat-rate drift is *not* in the LP).
Write LP(μ) for this problem and x*(μ) for an optimal solution.

## (i) Fleet monotonicity

Let μ1 < μ2, x1 = x*(μ1), x2 = x*(μ2). Optimality gives

    f(x1) + μ1 a(x1) ≤ f(x2) + μ1 a(x2)      and      f(x2) + μ2 a(x2) ≤ f(x1) + μ2 a(x1).

Adding, (μ2 − μ1)(a(x2) − a(x1)) ≤ 0, so a(x2) ≤ a(x1) for every pair of optimal selections. ∎
(Individual e_g need not be monotone: see the merit-order crossings below.)

## (ii) Lower edge

Let LP⁰ be LP(μ) with e fixed at e̲. Its optimal basis and dual (λ⁰, η⁰_R) do not depend on μ,
because μ multiplies only the (now fixed) e. Let x⁰ be a primal optimum of LP⁰.
In LP(μ), x⁰ is feasible. Its dual vector extends to the e-variables with reduced cost

    d_g(μ) = c_g + α_g μ − λ̃⁰_g,        λ̃⁰_g = λ⁰_{b(g)} − η⁰_R

(raising p_g by one MW at bus b(g) is worth the nodal price and consumes one MW of reserve headroom).
Every other variable has the same reduced cost as in LP⁰, which is ≥ 0 by optimality of LP⁰.
By LP optimality conditions, x⁰ is optimal for LP(μ) iff all d_g(μ) ≥ 0 for the e-variables at their lower
bounds, i.e. iff μ ≥ max_g (λ̃⁰_g − c_g)⁺ / α_g = μ̲_t. If every d_g > 0 (μ > μ̲_t), complementary
slackness forces e_g = e̲_g in *every* optimum, so the optimal e is unique. If LP⁰ is dual-degenerate,
d_g depends on which dual optimum is used; x⁰ is optimal iff *some* dual optimum gives d ≥ 0, so
μ̲_t is the smallest value over the dual optimal face. ∎

## (iii) Upper shedding ceilings

Fix only unit g at e̲_g and price all other units at μ. Suppose this restricted LP sheds load s_{b'} in the
interior of [0, 0.3 d_{b'}], at a bus b' not separated from b(g) by a binding line, with the reserve
constraint slack. Interior shedding means the nodal price at b' equals VOLL; with no binding line between b'
and b(g), λ_{b(g)} = VOLL, and with slack reserve η_R = 0, so λ̃_g = VOLL. The reduced cost of e_g in
this LP is c_g + α_g μ − VOLL, which is ≥ 0 iff μ ≥ (VOLL − c_g)/α_g = μ_g^hi.
Hence below μ_g^hi unit g's excess displaces shedding; above it, shedding displaces the excess. ∎

## (iv) Flat band: what is proved and what is checked

**Lemma (parametric LP).** For fixed Ω_t, the map μ ↦ optimal basis of LP(μ) is piecewise constant with
finitely many breakpoints; on each open interval between consecutive breakpoints, the optimal e (and, with
a deterministic tie-break among optima of the remaining LP, the whole dispatch) is constant.
(Standard sensitivity analysis: a basis stays optimal while all reduced costs keep their sign, and reduced
costs are affine in μ.)

Let B_t ⊂ (0, ∞) be the breakpoint set of step t. (ii) identifies the *last* discretionary breakpoint μ̲_t;
the merit-order crossings μ_{g,h} = (c_g − c_h)/(α_h − α_g) are breakpoints when no network constraint
separates g and h; (iii) identifies the ceilings of single-unit indispensable units.

**Completeness condition (C).** Along a trajectory ξ, every breakpoint of every B_t lies outside the open
interval (L(ξ), U(ξ)), where L is the maximum of the μ̲_t on discretionary steps and of the crossings among
indispensable units, and U is the minimum of the ceilings.

**Theorem (flatness under (C)).** Let μ ∈ (L(ξ(μ)), U(ξ(μ))) and assume (C) for ξ = ξ(μ). Then every
μ' ∈ (L, U) generates the same trajectory, so J(μ') = J(μ), and (L(ξ(μ')), U(ξ(μ'))) = (L, U).

*Proof.* Induct on t. If the states agree at t, μ and μ' lie in the same open interval of R \ B_t (no
breakpoint of B_t lies in (L, U) by (C)), so the dispatch coincides by the Lemma; the next state is a function
of the dispatch and the exogenous draw, hence agrees. Equal trajectories give equal L, U. ∎

(C) is what the single-unit formulas cannot guarantee. Two substitutable units that can each replace all
shedding when the other is available are *not* indispensable, yet their joint replacement by shedding is a
breakpoint absent from U. We therefore (a) state (C) as an explicit hypothesis and (b) verify it numerically:
`experiments/r18_breakpoint_audit.py` sweeps μ on 50 harvested states, finds all breakpoints, and reports none
inside the claimed band and no shedding onset on the 30 states whose LP⁰ is slack. This is evidence for the
5-bus fleet, not a proof for general fleets.

## Optimality: the dynamic assumption (A)

Let V_{t+1}(s) be the cost-to-go. If V_{t+1} is convex and differentiable in the wear direction, the optimal
action at (t, s_t) solves LP(γ_t) with γ_t = −∂V_{t+1}/∂(wear units) (a state-dependent price). **(A′):** for
every t on the optimal trajectory ξ*, γ_t lies in the same open interval of R \ B_t as the band (L, U).
Under (A′) the optimal action at every step equals the action at any constant μ in the band (Lemma), so the
constant price reproduces the optimal trajectory and is optimal: argmin J ⊇ (L, U), with equality up to
endpoint ties when J is strictly larger outside. Two caveats, both stated in the paper: (1) convexity of V_{t+1}
is not guaranteed because the *true* cost includes heat-rate drift (bilinear in c_g · hr(W_g) · p_g), which
the dispatch LP does not model; (2) (A′) is a statement about the unknown V, so we test it
(`experiments/r21_assumption_A.py`, one-step deviations from the certificate price) rather than assume it.
(A′) fails at long memory (τ = 48 h, `r17`), where the best constant lies above μ_hi.

## (C) for general fleets: a verifiable flat interval (unconditional)

**Theorem 2.** Let φ_t(μ) = min_{x∈Ω_t} f(x) + μ a(x). Then φ_t is concave (a pointwise minimum of affine
functions of μ). If a dispatch x̂ ∈ Ω_t is optimal at μ₁ < μ₂, it is optimal at every μ ∈ [μ₁, μ₂].

*Proof.* ℓ(μ) = f(x̂) + μ a(x̂) is affine and ℓ ≥ φ_t everywhere (x̂ is feasible), with ℓ(μ₁) = φ_t(μ₁),
ℓ(μ₂) = φ_t(μ₂). Concavity gives φ_t(μ) ≥ the chord of φ_t between μ₁ and μ₂, which equals ℓ(μ). So
φ_t = ℓ on [μ₁, μ₂], i.e. x̂ is optimal there. ∎

**Corollary (no assumption on the fleet).** Let ξ be the certificate trajectory with dispatches x_t at price μ_c,
and I_t = {μ : x_t optimal for LP_t(μ)}. By Theorem 2 each I_t is an interval containing μ_c; its endpoints
are found by bisection, each probe being one LP solve and a cost comparison φ_t(μ) = f(x_t) + μ a(x_t)
(two probes per step suffice to *verify* a candidate interval). Put I = ∩_t I_t. A policy that plays x_t at
every step is an optimal response to every μ ∈ I, so every μ ∈ I yields the trajectory ξ and the same J
(exactly, if the optimal dispatch is unique, which holds generically; otherwise for the tie-break that selects x_t).
Condition (C) of the theorem above is then equivalent to (L(ξ), U(ξ)) ⊆ I, and **this is checkable
without any single-unit formula**. If the check fails, I is the correct (smaller) band.

**Why the closed forms alone are not enough.** Two units that can each cover a shortfall are not
"indispensable", so U = ∞ and L ignores their merit-order switch. Example (`r24_completeness.py`): c = (20, 30),
α = (0.01, 0.02), 20 MW above the free blocks. The closed form gives (L, U) = (0, ∞), but the dispatch changes at
μ = (30−20)/0.01 = 1000 (verified interval at μ = 10³: [10⁻³, 1000.009]) and at the shedding switch. On 391
random fleets (2–8 units, random costs, aging rates, ramp windows and loads) the closed-form band was
not flat in 15.3% of cases, while the verified interval was unsound in 0 cases.

## Convexity of the cost-to-go

(A′) needs V_{t+1} convex in the wear direction. `experiments/r26_convexity.py` evaluates the best-of-grid
cost-to-go (5 prices, 2-day horizon, true simulator) on lines W₀ + θd, θ ∈ {0, .08, .16, .24, .32}, at every
96th step of certificate trajectories (3 seeds, `peak`, `tight 1.15`; 168 lines per variant). With the default
heat-rate drift, 40/42 (`peak`) and 42/42 (`tight 1.15`) lines are convex (worst second difference −307 $,
0.017% of V). With the heat-rate channel switched off, 38/42 and 42/42 (worst −783 $, 0.043%). So heat-rate drift
did not create the non-convexity in this test; the few violations are small and coincide with price switches of
the best-of-grid minimum. This is evidence, not a proof of convexity of V*.
