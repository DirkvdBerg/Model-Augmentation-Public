# Merged optimisation proposals for `Clean/` (2026-10-21)

Inputs: `optimise-claude.md` (IDs A, B, C, X; about -560 lines) and `optimise-codex.md` (IDs P, Q, X01; about -93
lines, counted conservatively). Codex's list is almost entirely contained in Claude's; the overlap is given per row.
Claude's IDs are the primary key below. Every item keeps bit-for-bit results by the constraints of the two
handoffs; the harness (layers 1 to 3c, with the H2 freshness check) re-proves the implemented set.

## A. Decisions for Dirk (recommendation first; all ten accepted by Dirk 2026-10-21, D-281)

| # | Item | Recommendation | Reason |
|-|-|-|-|
| O1 | B6: drop the `optimizer` property (reopens D-275 D1) | **Reject** | D1 stands; Codex keeps it for dump semantics; -7 lines is not worth reopening a ruling |
| O2 | A6: drop the two `least_squares_start` self-checks | **Reject** | they check a solver result (rule G keeps solver guards); the Codex judge called them justified |
| O3 | A5: drop `_json_safe` | **Reject** | medium risk, a cheap guard against a numpy scalar breaking the final `.npz` save |
| O4 | C22: drop the PS2 `.grad` save and restore | **Reject** | correct only by an argument about torch defaults; -3 lines for a medium risk |
| O5 | B33: drop the KeyboardInterrupt handler | **Reject** | changes Ctrl-C behaviour (no end-of-fit save) |
| O6 | C45 and C28: remove or slim `fixed_System_data` | **Reject** | loses the stride-1 float32 protection (design H5); Codex advises keeping the constructor |
| O7 | C36: update two `reason` strings naming removed options | **Accept** | text only, on the OOM and no-validation paths; not compared by any local layer |
| O8 | Items that need a harness edit (B5, B41, B51, C3, C19b) | **Accept, with the harness edit** | the harness is ours; each edit only follows a renamed or dropped field |
| O9 | All "breaks choice 3" items (Claude's list, Codex X01) | **Reject** | small gains; they cost the 1:1 diff against mainline and, for moved classes, existing dumps; Codex P06/P07 give most of the interconnect gain inside choice 3 |
| O10 | Items that change the compiled graph (B1 = P01, B2 = P02, B26) | **Accept** | identical on CPU (layers 1 to 3c); the compiled path is proven by the 6.4 GPU smokes at the end. If a smoke fails there, these three are the first to revert |

## B. Accepted without decision (low risk, all within choice 3)

**Largest:** A1 (`resolve_resume_hp`, -30), A2 = P04 extended (derived config values as ClassVars, -19), B1 = P01
(-17), A3 (-14), A4 (-12), B2 = P02 (-12), B3 = P08 (-12), C1 = Q04 (-12), C2 (-9), C3 (-9, harness edit), B4 (-8),
B5 (-8, harness edit), C4 = P11 (-8), C5 (-8), A7 (-7), C6 to C9 (C9 = P10) (-7 each), X1 (-7).

**The rest of Claude's ranked list**, A8 to A45, B7 to B71, C10 to C44, X2 to X5, except the items rejected in A.
Codex equivalents: P03 = C19 + C32, P05 = B7, P06 = B13, P07 (Codex's in-choice-3 parse simplification, taken
instead of Claude's choice-3 alternative), P09 = B17, P12 = B40, P13 = B37, P14 = B35, P15 = B10 + B45, P16 = A21,
P17 and P27 = A15, P18 = C27, P19 = C11, P20 = C21, P21 = part of B14, P23 = B50, P24 = C13, P25 = C10, P26 = A27,
P28 = A31, Q01 = B52 group, Q03 = A44, Q07 = B14.

**Codex only, accepted:** P22 (return the weighted loss directly, -1), P29 (`connected_blocks[0]` at OBC attachment,
-1), Q02 (call `tangent_pair` directly), Q05 (simplified PS2 screen condition), Q06 (unpack `r` directly).

**Estimated total:** about -460 lines (about 10 % of 4,506), after removing the rejected items.

## C. Not code changes, kept for later

- Stale text found by Claude (`optimise-claude.md`, last section): goes to the comment pass.
- The latent 3-D state `UnboundLocalError` in `interconnect.py`: removed by B19 with the flag itself.
- Codex's warning: savings and safety are reassessed during implementation; a proposal that fails a layer bitwise is
  reverted and recorded, never toleranced.
