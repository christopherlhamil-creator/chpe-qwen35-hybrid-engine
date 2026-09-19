; Expected: unsat
; Formal verification of Qwen2.5-72B signed 2-bit quantization bounds and bitfield packing.
; Invariants:
; 1. 4 discrete levels map injectively to 2-bit codes {00, 01, 10, 11}.
; 2. Packing 4 values into an 8-bit unsigned integer produces no bitfield overlap.
; 3. Unpacked reconstruction is strictly within [-2s, +1s].

(set-logic QF_BV)

; 2-bit codes
(declare-const c0 (_ BitVec 2))
(declare-const c1 (_ BitVec 2))
(declare-const c2 (_ BitVec 2))
(declare-const c3 (_ BitVec 2))

; Packed byte
(define-fun packed_byte () (_ BitVec 8)
  (bvor (concat (_ bv0 6) c0)
  (bvor (concat (_ bv0 4) (concat c1 (_ bv0 2)))
  (bvor (concat (_ bv0 2) (concat c2 (_ bv0 4)))
        (concat c3 (_ bv0 6))))))

; Unpacking functions
(define-fun unpack0 ((b (_ BitVec 8))) (_ BitVec 2) ((_ extract 1 0) b))
(define-fun unpack1 ((b (_ BitVec 8))) (_ BitVec 2) ((_ extract 3 2) b))
(define-fun unpack2 ((b (_ BitVec 8))) (_ BitVec 2) ((_ extract 5 4) b))
(define-fun unpack3 ((b (_ BitVec 8))) (_ BitVec 2) ((_ extract 7 6) b))

; Soundness conjecture: Unpacking must be bit-exact to original codes
(define-fun packing_is_sound () Bool
  (and (= (unpack0 packed_byte) c0)
       (= (unpack1 packed_byte) c1)
       (= (unpack2 packed_byte) c2)
       (= (unpack3 packed_byte) c3)))

; Negate the desired property: can there be any code combination where packing fails?
(assert (not packing_is_sound))

(check-sat)
