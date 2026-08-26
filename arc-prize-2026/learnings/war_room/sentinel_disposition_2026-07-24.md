# Sentinel line disposition memo — 2026-07-24

Filed per R19 methodology MAJOR ("either run the $0 pre-registered W2 instrument or
write a disposition memo stating the evidence weights and their dependence structure
before shelving") and R19 rl-planning MINOR (W2 = zero-VOI; slot 2 goes to tr87).
This memo takes the second branch. The R19 ruling context: the arm is HARM-PAUSED
(A21/C2, sealed pre-observation) and cannot re-enter the LB without a new A21 entry
case regardless of W2's outcome — so W2's result changes no live decision.

## Disposition: SHELVED — "certified observable, no lift channel"

The sentinel v2 patch remains available as passive telemetry for future arms (≈0
token cost when not crossing). No W3. No further sentinel windows. The W1 mechanism
seal STANDS (that half was cleanly earned: 22 sidecars + 56 stdout events exact,
once-per-game keying proven live).

## Evidence weights and dependence structure (stated, per methodology)

Three negative observations, TWO dependence clusters — not "three independent
signals" (the brief's phrase is hereby corrected):

1. **Eval-rail cluster (one rail, two replicate seeds, same build/composition):**
   Δlog1p(RHAE) s1 −0.315 (p=0.997), s2 −0.166 (p=0.90) vs null10. These are
   replicates within one rail and count as ONE corroborated rail-level signal,
   not two. No formal combination rule was pre-registered; none is applied. The
   rail-level statement is qualitative: both seeds negative, neither near zero.
2. **Scored-rail cluster (one draw, different rail, shared composition):** LB 0.71,
   z = −1.70 vs frozen control under plain normal; under the honest t-predictive
   (ν=9, √(1+1/10) inflation, per R19 methodology) t ≈ −1.62, one-sided p ≈ 0.07.
   Suggestive only. A healthy arm draws <0.80 with p ≈ 13% under the pooled
   posterior — the pause alone is consistent with noise, and no inference is
   claimed from it (C2).

The two clusters share the composition but nothing else (different rails, different
metrics, different regimes: capped-150 eval vs uncapped scored). The shelving
decision therefore rests on: rail-1 corroborated negative + sealed W1 behavioral
verdict ("fires, doesn't pay": 21/22 fired games kept grinding; +618 total actions)
+ doctrinal zero-upside under the completion-weighted scorer (efficiency observable,
no depth channel) + rail-2 consistency. The scored draw is CONSISTENT-WITH, not
load-bearing; the sealed eval evidence is load-bearing.

## Why not W2

W2's calibrated instrument (z-rule vs w0_s1=1.731, two-seed KILL α≈0.02) remains
valid and unspent. It is not run because its outcome is decision-inert: the arm is
paused, cannot re-enter without a new entry case, and the line's shelving does not
claim the formal KILL — it claims "no demonstrated lift channel + harm-paused,"
which the existing sealed evidence supports without a new window. If any future
panel wants the formal KILL on the record, W2 stays pre-registered and $0; running
it then costs a push slot, nothing else. Precedent note (methodology's concern):
this does NOT establish that sealed lines close by informal aggregation — it
establishes that a PAUSED arm with decision-inert confirmatory instruments may be
shelved by memo, with the instrument left armed.

## Bookkeeping

- Exploration budget: 1/12 spent; 11 remain. The window bought: live validation of
  the A21 harm-pause machinery end-to-end (entry → draw → pause → disposition) and
  the scored-rail confirmation of the eval-rail prediction. R19's critique that the
  entry bar admitted a known-null arm is ACCEPTED — entry-bar amendment (positive
  right-tail evidence required) is queued for the next amendment file, before
  draw 2/12.
- LB best unchanged 1.33. Frozen control ledger unchanged (0.71 excluded, different
  composition).
