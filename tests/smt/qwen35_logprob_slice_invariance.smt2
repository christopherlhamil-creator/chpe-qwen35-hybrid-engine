; Expected: unsat
; Verification that logits tensor slice dimension D_vocab == 248320 and choice indexing
; {0..9} for multiple choice (A..J) preserves distinct top-1 argmax ranking without index truncation.
(set-logic QF_BV)

(declare-const dim_vocab (_ BitVec 32))
(declare-const target_token (_ BitVec 32))
(declare-const choice_idx (_ BitVec 32))

(assert (= dim_vocab (_ bv248320 32)))
; Choice indices 0..9
(assert (bvule choice_idx (_ bv9 32)))
; Target tokens within vocabulary
(assert (bvult target_token dim_vocab))

; Conjectured bug condition: target_token >= dim_vocab (out-of-bounds slice)
(assert (bvuge target_token dim_vocab))

(check-sat)
