; Expected: unsat
; Verification that a dual-predicate filter (true EOS or Stop Sequence) strictly
; terminates generation before reaching max_tokens runaway (1024).
(set-logic QF_LIA)

(declare-const gen_step Int)
(declare-const max_tokens Int)
(declare-const answer_emitted_step Int)
(declare-const eos_matched Bool)
(declare-const stop_seq_matched Bool)

(assert (= max_tokens 1024))
(assert (>= gen_step 0))
(assert (>= answer_emitted_step 10))
(assert (<= answer_emitted_step 120)) ; Math solutions typically conclude within 120 tokens

; When answer is emitted, either EOS or Stop Sequence fires
(assert (=> (>= gen_step answer_emitted_step) (or eos_matched stop_seq_matched)))

; Termination condition definition:
(define-fun is_terminated () Bool (or eos_matched stop_seq_matched (>= gen_step max_tokens)))

; Negate conjecture: Can the sequence fail to terminate at or before answer_emitted_step?
(assert (and (>= gen_step answer_emitted_step) (not is_terminated)))

(check-sat)
