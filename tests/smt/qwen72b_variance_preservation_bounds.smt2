; Expected: unsat
; Formal verification of EBM Variance Preservation Bounds for deep 80-layer stacks.

(set-logic QF_LRA)

(declare-const alpha Real)
(declare-const delta Real)

; Scale factor compensation bounds
(assert (>= alpha 0.90))
(assert (<= alpha 1.20))

; Under EBM calibration, per-layer error delta is tightly bounded: delta <= 0.0005
(assert (>= delta 0.0))
(assert (<= delta 0.0005))

; Linearized cumulative deviation: 80 * delta <= 0.04 < 0.05
; Conjectured bug: cumulative deviation exceeds 5% bound
(assert (> (* 80.0 delta) 0.04))

(check-sat)
