; Expected: unsat
; Formal verification that Qwen3.5 EOS tokens (248044, 248046) fall strictly within
; the 248320 vocabulary boundary and are strictly distinct from legacy 151643.
(set-logic QF_BV)

(define-fun vocab_size () (_ BitVec 32) (_ bv248320 32))
(define-fun eos_endoftext () (_ BitVec 32) (_ bv248044 32))
(define-fun eos_im_end () (_ BitVec 32) (_ bv248046 32))
(define-fun legacy_qwen2_eot () (_ BitVec 32) (_ bv151643 32))

; Invariant: Both true EOS tokens are strictly inside vocab
(define-fun in_vocab ((t (_ BitVec 32))) Bool (bvult t vocab_size))

; Invariant: Legacy EOT is NOT equal to true Qwen3.5 EOS
(define-fun distinct_from_legacy () Bool 
    (and (distinct eos_endoftext legacy_qwen2_eot)
         (distinct eos_im_end legacy_qwen2_eot)))

; Negate the desired soundness invariant to test unsatisfiability
(assert (not (and (in_vocab eos_endoftext)
                  (in_vocab eos_im_end)
                  distinct_from_legacy)))

(check-sat)
