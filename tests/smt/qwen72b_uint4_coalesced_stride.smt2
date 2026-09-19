; Expected: unsat
; Formal verification of Qwen2.5-72B memory layout, 128-bit uint4 warp coalescing,
; and sector-aligned stride invariances.

(set-logic QF_NIA)

(define-fun hidden_dim () Int 8192)
(define-fun inter_dim () Int 29568)
(define-fun uint4_bytes () Int 16)
(define-fun half_cache_line_bytes () Int 32)
(define-fun cache_line_bytes () Int 64)
(define-fun warp_txn_bytes () Int 512)
(define-fun cell_bytes () Int 17408)
(define-fun prefetch_bytes () Int 3072)
(define-fun sector_bytes () Int 20480)

(define-fun hidden_packed_row_bytes () Int (div hidden_dim 4)) ; 2048 bytes
(define-fun inter_packed_row_bytes () Int (div inter_dim 4))   ; 7392 bytes

; Invariant 1: sector geometry equals cell + prefetch
(define-fun sector_geometry_sound () Bool
  (= sector_bytes (+ cell_bytes prefetch_bytes)))

; Invariant 2: hidden packed row is exactly divisible by uint4, cache_line, and warp transaction (512B)
(define-fun hidden_alignment_sound () Bool
  (and (= (mod hidden_packed_row_bytes uint4_bytes) 0)
       (= (mod hidden_packed_row_bytes cache_line_bytes) 0)
       (= (mod hidden_packed_row_bytes warp_txn_bytes) 0)))

; Invariant 3: intermediate packed row is exactly divisible by uint4 (16B) and half-cache-line (32B)
(define-fun inter_alignment_sound () Bool
  (and (= (mod inter_packed_row_bytes uint4_bytes) 0)
       (= (mod inter_packed_row_bytes half_cache_line_bytes) 0)))

; Soundness conjecture: all alignments and geometries must hold
(define-fun all_invariants_hold () Bool
  (and sector_geometry_sound
       hidden_alignment_sound
       inter_alignment_sound))

; Negate the invariant to test unsatisfiability
(assert (not all_invariants_hold))

(check-sat)
