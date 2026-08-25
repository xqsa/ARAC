# ARAC-OC Recovery-First Protocol

`recovery_first_protocol_v1.json` freezes AOB-24, seeds 117–141, Phase-I
180,000 FE, terminal 3,000,000 FE, historical family mapping, boundary
profile, and the read-only checkpoint/vendor sources. Patch, soft routing and
new selector are disabled.

## Gate order

1. B0 checks every checkpoint/current receipt binding and source hash.
2. B1 executes all 24 × 25 × 4 fixed-action arms. Partial or representative
   lanes are reported as incomplete. Aggregate historical tables without the
   original per-action seed metadata are marked inferred, not bitwise recovered.
3. B2 checks selector input hash → output hash → action without evaluating an
   action.
4. B3 checks Phase-I → selector → selected action terminal contracts per case.

B1 failure stops innovation work. The default production anchor remains the
recovered historical baseline until every required gate passes.

## B1-Screen Result

The first reduced screen used 5 seeds (`117, 123, 129, 135, 141`) and completed
all 120 mapped-action arms with zero contract failures. The displayed-mean screen
did not pass: AOR passed 4/6 cases, SMP 1/6, GCB 0/6, and CTP 5/6. This is a
diagnostic recovery failure, not evidence against shared-patch. B1-Final and
innovation experiments remain blocked until the action-specific regression is
isolated.
