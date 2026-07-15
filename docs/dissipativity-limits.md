# Dissipativity / Passivity Restrictions (consolidated reference)

**Date**: 2026-07-11. **Purpose**: one authoritative list of every restriction of the dissipativity /
passivity / NI family for the X/Y free-integrator augmentation, why each holds, and where it is treated in
context. Scattered originally across `drift-diagnosis-status.md` (§5f, §5j, §5m), `passivity-augmentation-
literature.md` (§G, §H), `dissipative-block-spec.md`, and `decisions.md` (D-104, D-106); this doc indexes
them. Quotes are transcribed from on-disk PDF text layers (primary-read where tagged); re-verify
character-exact before thesis use.

**Two kinds of restriction.**
- **A. Boundedness limits** — what dissipativity CANNOT GUARANTEE (it does not bound free-integrator
  position, or its guarantee needs a premise we lack).
- **B. Expressivity limits** — what dissipativity FORBIDS the ANN from learning (it restricts the class of
  representable dynamics; fatal for the unknown-system deliverable, which requires full expressivity).

The X/Y axes are zero-stiffness mass-dampers: velocity is damped (finite tau_X=1.55 s, tau_Y=1.01 s) but
POSITION is a free integrator (pole at the origin). Every limit below is about that free-integrator pole.

---

## A. Boundedness limits (what it cannot guarantee)

### A1. Passivity bounds VELOCITY, not POSITION (the O(sqrt(T)) limit)
**Statement.** Passivity (`int F.v <= V(0)`, `V >= 0`) bounds kinetic energy / velocity (`v in L2`), NOT
position on a free integrator. By Cauchy-Schwarz `|q(T)-q(0)| = |int v| <= sqrt(T) sqrt(int v^2) = O(sqrt(T))`,
sub-linear but unbounded as `T -> inf`. Position is bounded IFF the net impulse `int F dt` is bounded, which
passivity does not force.
**Why.** For the mass-damper, `F->v = 1/(ms+c)` is BIBO-stable (damping bounds velocity), but `F->q =
1/(s(ms+c))` keeps the pole at the origin; a bounded-energy near-DC force is invisible to `int F.v` at low
velocity yet walks the integrator.
**Where.** `drift-diagnosis-status.md` §5j (measured: stored-energy probe, env_ratio 1.62); independently
re-derived and CONFIRMED in `passivity-augmentation-literature.md` §G10.

### A2. Cyclo-dissipativity (indefinite-storage relaxation) gives "only instability results"
**Statement.** Relaxing storage to be indefinite (the marginal case) does NOT yield a boundedness/Lyapunov
conclusion; it supports instability theorems, not stability.
**Quote [PRIMARY-READ: van der Schaft, "Cyclo-dissipativity revisited", arXiv:2003.10143, Remark 3.4, p.8].**
> "the Lyapunov function ... is no longer nonnegative. Hence in principle only instability results can be
> inferred; this is the motivation for cyclo-dissipativity."
Storage is explicitly allowed indefinite (p.6 footnote: "we do not yet require S to be nonnegative or
bounded from below").
**Where.** `passivity-augmentation-literature.md` §H1; `drift-diagnosis-status.md` §5m; `decisions.md` D-106.

### A3. Equilibrium-Independent / shifted / Krasovskii passivity characterize the marginal mode but do not bound POSITION
**Statement.** EID/EIP (passivity w.r.t. any equilibrium, continuum-of-equilibria) and shifted/Krasovskii
passivity characterize systems with a free/continuum equilibrium, but bound shifted input-output behaviour,
not the free coordinate. A mass-damper IS EID (velocity-passive around any position) yet position still
integrates.
**Quote [PRIMARY-READ: Simpson-Porco, arXiv:1709.06986, Def 3.2, p.3-4].** EID requires "for every
equilibrium x-bar ... a storage function V_x-bar : X -> R>=0 with V_x-bar(x-bar)=0" and the shifted supply
rate; the assignable-equilibria set is all of X when m=n (every state an equilibrium = the free integrator).
**Where.** `passivity-augmentation-literature.md` §H2 (EID), §H4 (Krasovskii); `drift-diagnosis-status.md` §5m.

### A4. ISS-based dissipative augmentation requires an ISS baseline (excludes the free integrator)
**Statement.** The state-of-the-art skew-dissipative residual (DiLaR-PINN, = our PH block) preserves
stability only IF the nominal model is ISS. A free integrator is not 0-GAS, so the premise fails and no
guarantee applies to X/Y.
**Quote [PRIMARY-READ: Long, Solak, Ajoudani, DiLaR-PINN, arXiv:2604.18277, Prop 3, p.4].** The augmented
system is ISS "Assume the nominal physical model ... admits an ISS-Lyapunov function V ... with
`grad V^T f_phys <= -alpha3(||x||) + sigma(||u||)`" (class-K-infinity alpha3). The free integrator admits no
such V on the position coordinate.
**Where.** `passivity-augmentation-literature.md` §G1; `drift-diagnosis-status.md` §5L.

### A5. Negative-Imaginary bounds POSITION only via an interconnection partner (open-loop free-run has none)
**Statement.** NI is the force->position passive class and CAN bound position, but the classical
position-boundedness comes from the feedback INTERCONNECTION (NI plant with a strictly-NI / OSNI partner,
DC-gain / residue condition). In OPEN-LOOP free-run (our deliverable metric) there is no partner, so
NI-of-the-block alone does not bound open-loop position.
**Why (counterexample).** A pure double integrator `1/s^2` is itself an NI system (Mabrok 2014 lists single/
double integrators as free-body NI), yet under a constant force it ramps to infinity. So "the block is NI"
cannot by itself stop a DC-driven drift; the object that drifts already satisfies NI.
**Where.** Established in discussion (open-loop vs closed-loop, "controller later"); classical NI free-body
is `passivity-augmentation-literature.md` §C1/§G3 (Mabrok, LTI-only) and §G8 (nonlinear NI, Shi-Petersen-
Vladimirov). On real CLOSED-LOOP Telica data the servo IS the OSNI partner, so NI is the right closed-loop
certificate there; it is the OPEN-LOOP free-run where it does not bind.

---

## B. Expressivity limits (what it forbids the ANN from learning)

### B1. Pure dissipativity forbids energy STORAGE (cannot represent the absorber)
**Statement.** Pure dissipativity (`int F.v <= 0`, a damper) forbids storing and returning energy, so it
cannot represent springs/masses / resonant modes. The hidden absorber is a mass-spring-damper (stores in its
spring and mass, returns it), so pure dissipativity walls it off.
**Where.** `drift-diagnosis-status.md` §5f (the table at 5f: "STORE + return energy ... FORBIDDEN under pure
dissipative"). This is why the project uses PASSIVITY (store + return), not pure dissipativity, wherever a
power-based constraint is considered.

### B2. Net-impulse / bounded-integral (impulse-based cousin) forbids dissipative FRICTION
**Statement.** The net-impulse / exact-difference output `g_k = psi(z_k) - psi(z_{k-1})` bounds accumulated
impulse `Sum g = psi(z_N) - psi(z_0)` for all weights, forbidding ANY non-conservative net-impulse force.
It is IMPULSE-based, blind to the sign of power, so it forbids energy-removing friction (Coulomb over
asymmetric motion carries net impulse) along with energy-injecting drift. Fails criterion 2 (friction-
permitting); forces friction into `f_base`.
**Where.** `dissipative-block-spec.md` §2 (expressivity note) + §5; `tasks/lessons.md` (impulse-based
constraints suppress friction like the mean penalty). NB: net-impulse is not dissipativity per se, but the
same power-blind family; it bounds POSITION (unlike A1) at the cost of B2.

### B3. Contraction (RENs / Gyorok) DAMPS the marginal mode (destroys the zero-stiffness physics)
**Statement.** Contraction forces strictly-stable dynamics, which damps the free integrator into a
strictly-stable axis, destroying the K=0 physics (fails criterion 3, marginal-preservation).
**Quote [PRIMARY-READ: Revay, Wang, Manchester, RENs, arXiv:2104.05942].** Contraction is w.r.t. a strictly
positive-definite metric: Theorem 1 requires "P = P^T > 0"; Definition 2 defines contraction with rate
"alpha in (0,1)" (strict); incremental passivity is enforced JOINTLY with contraction, no P>=0 / marginal
variant. Gyorok 2026 (arXiv:2604.11421) likewise forces `||A|| < 1` (VERIFIED).
**Where.** `passivity-augmentation-literature.md` §G2, catalog A4; `drift-diagnosis-status.md` §5L.

### B4. Energy-Casimir control DAMPS the mode (and is a controller, not a forward augmentation)
**Statement.** The Casimir concept (a flat/conserved storage direction) is marginal-relevant, but its
deployment in energy-Casimir CONTROL adds damping to asymptotically stabilize a desired equilibrium (fails
marginal-preservation), and it is a controller, not a forward-model augmentation.
**Quote [PRIMARY-READ: Xu, Zakwan, Ferrari-Trecate, arXiv:2112.03339].** the method "asymptotically
stabilize[s] port-Hamiltonian systems at a desired equilibrium" and "an additional damping term is employed
to asymptotically stabilize the closed-loop system."
**Where.** `passivity-augmentation-literature.md` §H3.

### B5. THE META-LIMIT: passivity RESTRICTS THE MODEL CLASS -> may forbid the true residual on an unknown system
**Statement.** Any hard, for-all-weights passivity/dissipativity constraint forbids energy-injecting
(active-LOOKING) residuals. The physical machine is passive, but the RESIDUAL in the chosen coordinates need
not be, and on an UNKNOWN system this is UNVERIFIABLE, so the constraint risks excluding the true residual.
More generally: a for-all-weights no-drift guarantee and FULL expressivity are logically incompatible (a
universal approximator can represent a drifting force), so any structural guarantee necessarily forbids some
representable dynamics. This is the decisive limit that removes dissipativity/passivity from the DELIVERABLE.
**Where.** `data-silent-regularization-concept.md` §1; `augmentation-validation-design.md` §5 (the
"active-looking residual" library member); `tasks/lessons.md` (rule: hard-guarantee XOR full-expressivity).

---

## Synthesis: why these collectively rule dissipativity out as the DELIVERABLE
- **On boundedness (A):** plain passivity does not bound free-integrator position (A1); the marginal
  relaxations that admit the pole (cyclo, EID, A2-A3) explicitly do NOT bound it either; the guarantees that
  DO bound position need a premise we lack (an ISS baseline A4, or an interconnection partner A5, absent in
  open-loop free-run).
- **On expressivity (B):** the power-based constraints that could preserve the marginal mode either forbid
  storage (B1) or friction (B2) or damp the mode (B3-B4); and the umbrella limit (B5) is that ANY hard
  passivity constraint restricts the model class, which is forbidden for an unknown-system deliverable.
- **Conclusion.** Dissipativity/passivity/NI is the right CLOSED-LOOP certificate (A5, real Telica servo as
  the OSNI partner) and a legitimate SIM-phase or fallback tool, but it cannot be the expressivity-preserving
  open-loop deliverable. The no-drift for the unknown-system deliverable must come from the ESTIMATOR
  (data-silent regularization of the unexcited direction), not from a dissipativity constraint on the model
  class. See `data-silent-regularization-concept.md` and `augmentation-validation-design.md`.

## Provenance / primary-read status
- PRIMARY-READ this session: van der Schaft cyclo (2003.10143), Simpson-Porco EID (1709.06986), DiLaR-PINN
  (2604.18277), RENs (2104.05942), Xu-Zakwan-Ferrari-Trecate Casimir (2112.03339), Mabrok free-body NI
  (1305.1079), Shi-Petersen-Vladimirov nonlinear NI (2011.14610), Kawano-Kosaraju-Scherpen Krasovskii
  (1907.07420).
- A1/§5j math and A5 open-loop/counterexample: re-derived, not from a single paper (Cauchy-Schwarz; the
  `1/s^2`-is-NI counterexample). B5: our synthesis under the user's full-expressivity requirement.
- All quotes: re-verify character-exact against the PDF before thesis use.
