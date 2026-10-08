-- Proofs of the laws of laws/ie_laws.cpp (cpp-lean: IeLaws.lean).
-- One theorem X : Laws.X per law. src/tools/leancheck (cpp-lean) checks
-- the names and the statements and lists the axioms (no sorryAx).
import IeLemmas
open C
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

namespace Proof

-- 1. Index and memory safety

theorem grid_cover : Laws.grid_cover := by
  intro rows r; simp only [law_grid_cover, ie_gemv_ngroups, ie_rows_per_wg, ie_gemv_row]; bv_decide

theorem grid_unique : Laws.grid_unique := by
  intro g v h w; simp only [law_grid_unique, ie_gemv_row]; bv_decide

theorem lane_cover : Laws.lane_cover := by
  intro b; simp only [law_lane_cover, ie_blk_lane, ie_wave, ie_lane_blk, ie_blk_step]; bv_decide

theorem lane_unique : Laws.lane_unique := by
  intro l t m s; simp only [law_lane_unique, ie_lane_blk]; bv_decide

theorem act_addr : Laws.act_addr := by
  intro nb b v; simp only [law_act_addr]; bv_decide

theorem q4_addr : Laws.q4_addr := by
  intro rows nb r b w k
  unfold law_q4_addr
  cases hs : ie_sizes_ok rows nb <;> cases hr : BitVec.ult r rows <;> cases hb : BitVec.ult b nb <;>
    cases hw : BitVec.ult w 4#32 <;> cases hk : BitVec.ult k 16#32 <;>
    simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and, Bool.not_true, Bool.false_or]
  simp only [ie_sizes_ok, Bool.and_eq_true, BitVec.ult_iff_lt] at hs hr hb hw hk
  have ⟨q1, q2⟩ := rows_nb rows nb hs.1 hs.2
  have ⟨p1, p2⟩ := r_nb rows nb r hs.1 hs.2 hr
  simp only [ie_q4_dst_word, ie_q4_dst_scale, ie_q4_src_qbyte, ie_q4_src_blk, Bool.and_eq_true, BitVec.ult_iff_lt]
  have hb' : b.toNat < nb.toNat := hb
  have hw' : w.toNat < 4 := hw
  have hk' : k.toNat < 16 := hk
  simp only [BitVec.lt_def, BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_ofNat, p1, q1]
  generalize r.toNat * nb.toNat = pn at *; generalize rows.toNat * nb.toNat = qn at *
  simp (disch := omega) only [Nat.mod_eq_of_lt]
  omega

theorem q4_inverse : Laws.q4_inverse := by
  intro rows nb r b w
  unfold law_q4_inverse
  cases hs : ie_sizes_ok rows nb <;> cases hr : BitVec.ult r rows <;> cases hb : BitVec.ult b nb <;>
    cases hw : BitVec.ult w 4#32 <;>
    simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and, Bool.not_true, Bool.false_or]
  simp only [ie_sizes_ok, Bool.and_eq_true, BitVec.ult_iff_lt] at hs hr hb hw
  have ⟨q1, q2⟩ := rows_nb rows nb hs.1 hs.2
  have ⟨p1, p2⟩ := r_nb rows nb r hs.1 hs.2 hr
  have hb' : b.toNat < nb.toNat := hb
  have hw' : w.toNat < 4 := hw
  have hN : 0 < nb.toNat := by omega
  simp only [ie_q4_word_row, ie_q4_word_blk, ie_q4_word_w, ie_q4_dst_word, Bool.and_eq_true, beq_iff_eq]
  have es : (r * nb + b).toNat = b.toNat + nb.toNat * r.toNat := by
    rw [BitVec.toNat_add, p1, Nat.mod_eq_of_lt (by omega), Nat.mul_comm, Nat.add_comm]
  have e4 : ((r * nb + b) * 4#32 + w).toNat / 4 = b.toNat + nb.toNat * r.toNat := by
    simp only [BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_ofNat, es]
    rw [Nat.mul_comm nb.toNat] at *
    generalize r.toNat * nb.toNat = pn at *
    simp (disch := omega) only [Nat.mod_eq_of_lt]
    omega
  refine ⟨⟨?_, ?_⟩, ?_⟩ <;> apply BitVec.eq_of_toNat_eq
  · rw [BitVec.toNat_udiv, BitVec.toNat_udiv, BitVec.toNat_ofNat, e4, Nat.add_mul_div_left _ _ hN, Nat.div_eq_of_lt hb', Nat.zero_add]
  · rw [BitVec.toNat_umod, BitVec.toNat_udiv, BitVec.toNat_ofNat, e4, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hb']
  · simp only [BitVec.toNat_umod, BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_ofNat, es]
    rw [Nat.mul_comm nb.toNat] at *
    generalize r.toNat * nb.toNat = pn at *
    simp (disch := omega) only [Nat.mod_eq_of_lt]
    omega

theorem q4_onto : Laws.q4_onto := by
  intro rows nb i
  unfold law_q4_onto
  cases hs : ie_sizes_ok rows nb <;> cases h0 : BitVec.ult 0#32 nb <;> cases hi : BitVec.ult i (rows * nb * 4#32) <;>
    simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and, Bool.not_true, Bool.false_or]
  simp only [ie_sizes_ok, Bool.and_eq_true, BitVec.ult_iff_lt] at hs h0 hi
  have ⟨q1, q2⟩ := rows_nb rows nb hs.1 hs.2
  have hN : 0 < nb.toNat := h0
  have hN' : nb.toNat < 512 := hs.2
  have hi' : i.toNat < rows.toNat * nb.toNat * 4 := by
    have := BitVec.lt_def.mp hi
    simp only [BitVec.toNat_mul, q1, BitVec.toNat_ofNat] at this
    simp (disch := omega) only [Nat.mod_eq_of_lt] at this; exact this
  simp only [ie_q4_word_row, ie_q4_word_blk, ie_q4_word_w, ie_q4_dst_word, Bool.and_eq_true, beq_iff_eq, BitVec.ult_iff_lt]
  have hq : i.toNat / 4 < rows.toNat * nb.toNat := by omega
  have hd : i.toNat / 4 / nb.toNat < rows.toNat := (Nat.div_lt_iff_lt_mul hN).mpr hq
  have hm : i.toNat / 4 % nb.toNat < nb.toNat := Nat.mod_lt _ hN
  have hdm := Nat.div_add_mod (i.toNat / 4) nb.toNat
  have hle : nb.toNat * (i.toNat / 4 / nb.toNat) ≤ i.toNat / 4 := by omega
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · rw [BitVec.lt_def, BitVec.toNat_udiv, BitVec.toNat_udiv, BitVec.toNat_ofNat]; exact hd
  · rw [BitVec.lt_def, BitVec.toNat_umod, BitVec.toNat_udiv, BitVec.toNat_ofNat]; exact hm
  · apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_udiv, BitVec.toNat_umod, BitVec.toNat_ofNat]
    simp (disch := omega) only [Nat.mod_eq_of_lt (a := 4) (b := 2^32)]
    rw [Nat.mul_comm (i.toNat / 4 / nb.toNat)]
    generalize nb.toNat * (i.toNat / 4 / nb.toNat) = pn at *
    simp (disch := omega) only [Nat.mod_eq_of_lt]
    omega

-- 2. Memory plan

theorem plan_fits : Laws.plan_fits := by
  intro p i x
  unfold law_plan_fits
  cases hc : ie_plan_check p <;> cases hi : BitVec.ult i p.n <;> simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and]
  have h := (plan_buf p i hc (BitVec.ult_iff_lt.mp hi)).1
  revert h
  simp only [ie_buf_ok, ie_in_buf, ie_align_mask]
  generalize ie_off p i = o; generalize ie_size p i = s; generalize p.cap = c
  bv_decide

theorem plan_aligned : Laws.plan_aligned := by
  intro p i
  unfold law_plan_aligned
  cases hc : ie_plan_check p <;> cases hi : BitVec.ult i p.n <;> simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and, Bool.not_true, Bool.false_or]
  have h := (plan_buf p i hc (BitVec.ult_iff_lt.mp hi)).1
  revert h
  simp only [ie_buf_ok, ie_align_mask]
  generalize ie_off p i = o; generalize ie_size p i = s; generalize p.cap = c
  generalize ie_first p i = f; generalize ie_last p i = l
  bv_decide

theorem plan_no_clobber : Laws.plan_no_clobber := by
  intro p i j t x
  unfold law_plan_no_clobber
  cases hc : ie_plan_check p <;> cases hi : BitVec.ult i p.n <;> cases hj : BitVec.ult j p.n <;>
    cases e : (i != j) <;> simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and]
  have hi' := BitVec.ult_iff_lt.mp hi
  have hj' := BitVec.ult_iff_lt.mp hj
  have ne : i ≠ j := by simpa using e
  have bi := (plan_buf p i hc hi').1
  have bj := (plan_buf p j hc hj').1
  have s : ie_buf_sep p i j = true ∨ ie_buf_sep p j i = true := by
    have : i < j ∨ j < i := by bv_omega
    rcases this with h | h
    · exact Or.inl (plan_sep p i j hc hi' hj' h)
    · exact Or.inr (plan_sep p j i hc hj' hi' h)
  revert bi bj s
  simp only [ie_buf_ok, ie_align_mask, ie_buf_sep, ie_live, ie_in_buf]
  generalize ie_off p i = oi; generalize ie_size p i = si; generalize ie_off p j = oj; generalize ie_size p j = sj
  generalize ie_first p i = fi; generalize ie_last p i = li; generalize ie_first p j = fj; generalize ie_last p j = lj
  generalize p.cap = c
  bv_decide

-- 3. Reductions

theorem strided_sum : Laws.strided_sum := by
  intro x nb
  unfold law_strided_sum
  cases h : BitVec.ult nb 1073741824#32 <;> simp only [Bool.not_false, Bool.true_or, Bool.not_true, Bool.false_or, beq_iff_eq]
  have hN : nb.toNat < 2^30 := BitVec.ult_iff_lt.mp h
  rw [ie_strided_sum, ie_lanes_sum, lanes_go x nb hN _ _ _ (by decide) (by decide), ie_seq_sum,
    seq_go x nb _ _ _ (by simp)]
  exact lanes_sq _ _ nb.toNat 0 (by omega)

theorem wave_sum : Laws.wave_sum := by
  intro x
  simp only [law_wave_sum, ie_wave_sum, ie_tree, ie_seq_sum, BitVec.toNat_ofNat, Nat.reduceMod, beq_iff_eq]
  simp only [ie_seq_sum.go, ie_tree.go, BitVec.reduceULT, BitVec.reduceAdd, BitVec.reduceSub, Bool.cond_true, Bool.cond_false,
    BitVec.add_zero, BitVec.zero_add]
  have h1 : (32#32 >>> (1#32 : U32) : U32) = 16#32 := by decide
  have h2 : (32#32 >>> (2#32 : U32) : U32) = 8#32 := by decide
  have h3 : (32#32 >>> (3#32 : U32) : U32) = 4#32 := by decide
  have h4 : (32#32 >>> (4#32 : U32) : U32) = 2#32 := by decide
  have h5 : (32#32 >>> (5#32 : U32) : U32) = 1#32 := by decide
  simp only [h1, h2, h3, h4, h5, BitVec.reduceAdd]
  generalize x.load = f
  ac_rfl

-- 4. Quantization

theorem q4_nibble : Laws.q4_nibble := by
  intro q0 q1 q2 q3 j; simp only [law_q4_nibble, ie_q4_nib, ie_pack4, ie_q4_spec_nib]; bv_decide

theorem q4_word_of : Laws.q4_word_of := by
  intro j; simp only [law_q4_word_of, ie_q4_word_of]; bv_decide

theorem q4q8_word : Laws.q4q8_word := by
  intro wq alo ahi
  simp only [law_q4q8_word, ie_q4q8_word, ie_dot4_us]
  rw [lo_nib wq 0#32 (by decide), lo_nib wq 1#32 (by decide), lo_nib wq 2#32 (by decide), lo_nib wq 3#32 (by decide),
      hi_nib wq 0#32 (by decide), hi_nib wq 1#32 (by decide), hi_nib wq 2#32 (by decide), hi_nib wq 3#32 (by decide)]
  simp only [ie_sum_s8x4, ie_q4q8_word_spec, BitVec.reduceAdd, beq_iff_eq]
  generalize ie_q4_nib wq 0#32 = n0 at *
  generalize ie_q4_nib wq 1#32 = n1 at *
  generalize ie_q4_nib wq 2#32 = n2 at *
  generalize ie_q4_nib wq 3#32 = n3 at *
  generalize ie_q4_nib wq 16#32 = n4 at *
  generalize ie_q4_nib wq 17#32 = n5 at *
  generalize ie_q4_nib wq 18#32 = n6 at *
  generalize ie_q4_nib wq 19#32 = n7 at *
  generalize ie_sext8 (ie_byte alo 0#32) = a0 at *
  generalize ie_sext8 (ie_byte alo 1#32) = a1 at *
  generalize ie_sext8 (ie_byte alo 2#32) = a2 at *
  generalize ie_sext8 (ie_byte alo 3#32) = a3 at *
  generalize ie_sext8 (ie_byte ahi 0#32) = a4 at *
  generalize ie_sext8 (ie_byte ahi 1#32) = a5 at *
  generalize ie_sext8 (ie_byte ahi 2#32) = a6 at *
  generalize ie_sext8 (ie_byte ahi 3#32) = a7 at *
  grind

theorem q4q8_word_acc (acc wq alo ahi : U32) :
    ie_q4q8_word acc wq alo ahi = acc + ie_q4q8_word 0#32 wq alo ahi := by
  simp only [ie_q4q8_word, ie_dot4_us]; ac_rfl
theorem q4q8_word_eq (wq alo ahi : U32) :
    ie_q4q8_word 0#32 wq alo ahi = ie_q4q8_word_spec wq alo ahi + 8#32 * (ie_sum_s8x4 alo + ie_sum_s8x4 ahi) := by
  have h := q4q8_word wq alo ahi
  simp only [law_q4q8_word, beq_iff_eq] at h
  rw [← h, BitVec.sub_add_cancel]
theorem q4q8_block : Laws.q4q8_block := by
  intro qw qb aw ab
  simp only [law_q4q8_block, ie_q4q8_block, beq_iff_eq]
  rw [q4q8_word_acc _ (qw.load (qb + 3#32)), q4q8_word_acc _ (qw.load (qb + 2#32)), q4q8_word_acc _ (qw.load (qb + 1#32))]
  simp only [q4q8_word_eq, ie_q4q8_word_spec, ie_q8_sum, ie_q4q8_spec, BitVec.toNat_ofNat, Nat.reduceMod]
  simp only [ie_q4q8_spec.go, ie_q4q8_term, ie_q4_word_of, BitVec.reduceULT, BitVec.reduceAdd, BitVec.reduceSub,
    BitVec.reduceAnd, BitVec.reduceMul, BitVec.reduceHShiftRight, Bool.cond_true, Bool.cond_false, BitVec.add_zero, BitVec.zero_add]
  generalize Mem.load qw qb = w0
  generalize Mem.load qw (qb + 1#32) = w1
  generalize Mem.load qw (qb + 2#32) = w2
  generalize Mem.load qw (qb + 3#32) = w3
  simp only [nib_eq w1 4#32 0#32 (by decide), nib_eq w1 5#32 1#32 (by decide), nib_eq w1 6#32 2#32 (by decide), nib_eq w1 7#32 3#32 (by decide), nib_eq w1 8#32 0#32 (by decide), nib_eq w1 9#32 1#32 (by decide), nib_eq w1 10#32 2#32 (by decide), nib_eq w1 11#32 3#32 (by decide), nib_eq w1 12#32 0#32 (by decide), nib_eq w1 13#32 1#32 (by decide), nib_eq w1 14#32 2#32 (by decide), nib_eq w1 15#32 3#32 (by decide), nib_eq w1 20#32 16#32 (by decide), nib_eq w1 21#32 17#32 (by decide), nib_eq w1 22#32 18#32 (by decide), nib_eq w1 23#32 19#32 (by decide), nib_eq w1 24#32 16#32 (by decide), nib_eq w1 25#32 17#32 (by decide), nib_eq w1 26#32 18#32 (by decide), nib_eq w1 27#32 19#32 (by decide), nib_eq w1 28#32 16#32 (by decide), nib_eq w1 29#32 17#32 (by decide), nib_eq w1 30#32 18#32 (by decide), nib_eq w1 31#32 19#32 (by decide), nib_eq w2 4#32 0#32 (by decide), nib_eq w2 5#32 1#32 (by decide), nib_eq w2 6#32 2#32 (by decide), nib_eq w2 7#32 3#32 (by decide), nib_eq w2 8#32 0#32 (by decide), nib_eq w2 9#32 1#32 (by decide), nib_eq w2 10#32 2#32 (by decide), nib_eq w2 11#32 3#32 (by decide), nib_eq w2 12#32 0#32 (by decide), nib_eq w2 13#32 1#32 (by decide), nib_eq w2 14#32 2#32 (by decide), nib_eq w2 15#32 3#32 (by decide), nib_eq w2 20#32 16#32 (by decide), nib_eq w2 21#32 17#32 (by decide), nib_eq w2 22#32 18#32 (by decide), nib_eq w2 23#32 19#32 (by decide), nib_eq w2 24#32 16#32 (by decide), nib_eq w2 25#32 17#32 (by decide), nib_eq w2 26#32 18#32 (by decide), nib_eq w2 27#32 19#32 (by decide), nib_eq w2 28#32 16#32 (by decide), nib_eq w2 29#32 17#32 (by decide), nib_eq w2 30#32 18#32 (by decide), nib_eq w2 31#32 19#32 (by decide), nib_eq w3 4#32 0#32 (by decide), nib_eq w3 5#32 1#32 (by decide), nib_eq w3 6#32 2#32 (by decide), nib_eq w3 7#32 3#32 (by decide), nib_eq w3 8#32 0#32 (by decide), nib_eq w3 9#32 1#32 (by decide), nib_eq w3 10#32 2#32 (by decide), nib_eq w3 11#32 3#32 (by decide), nib_eq w3 12#32 0#32 (by decide), nib_eq w3 13#32 1#32 (by decide), nib_eq w3 14#32 2#32 (by decide), nib_eq w3 15#32 3#32 (by decide), nib_eq w3 20#32 16#32 (by decide), nib_eq w3 21#32 17#32 (by decide), nib_eq w3 22#32 18#32 (by decide), nib_eq w3 23#32 19#32 (by decide), nib_eq w3 24#32 16#32 (by decide), nib_eq w3 25#32 17#32 (by decide), nib_eq w3 26#32 18#32 (by decide), nib_eq w3 27#32 19#32 (by decide), nib_eq w3 28#32 16#32 (by decide), nib_eq w3 29#32 17#32 (by decide), nib_eq w3 30#32 18#32 (by decide), nib_eq w3 31#32 19#32 (by decide)]
  generalize (ie_q4_nib w0 0#32 - 8#32) * ie_sext8 (ie_byte (aw.load ab) 0#32) = t0_0_0
  generalize (ie_q4_nib w0 1#32 - 8#32) * ie_sext8 (ie_byte (aw.load ab) 1#32) = t0_0_1
  generalize (ie_q4_nib w0 2#32 - 8#32) * ie_sext8 (ie_byte (aw.load ab) 2#32) = t0_0_2
  generalize (ie_q4_nib w0 3#32 - 8#32) * ie_sext8 (ie_byte (aw.load ab) 3#32) = t0_0_3
  generalize (ie_q4_nib w0 16#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 4#32)) 0#32) = t0_1_0
  generalize (ie_q4_nib w0 17#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 4#32)) 1#32) = t0_1_1
  generalize (ie_q4_nib w0 18#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 4#32)) 2#32) = t0_1_2
  generalize (ie_q4_nib w0 19#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 4#32)) 3#32) = t0_1_3
  generalize (ie_q4_nib w1 0#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 1#32)) 0#32) = t1_0_0
  generalize (ie_q4_nib w1 1#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 1#32)) 1#32) = t1_0_1
  generalize (ie_q4_nib w1 2#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 1#32)) 2#32) = t1_0_2
  generalize (ie_q4_nib w1 3#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 1#32)) 3#32) = t1_0_3
  generalize (ie_q4_nib w1 16#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 5#32)) 0#32) = t1_1_0
  generalize (ie_q4_nib w1 17#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 5#32)) 1#32) = t1_1_1
  generalize (ie_q4_nib w1 18#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 5#32)) 2#32) = t1_1_2
  generalize (ie_q4_nib w1 19#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 5#32)) 3#32) = t1_1_3
  generalize (ie_q4_nib w2 0#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 2#32)) 0#32) = t2_0_0
  generalize (ie_q4_nib w2 1#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 2#32)) 1#32) = t2_0_1
  generalize (ie_q4_nib w2 2#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 2#32)) 2#32) = t2_0_2
  generalize (ie_q4_nib w2 3#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 2#32)) 3#32) = t2_0_3
  generalize (ie_q4_nib w2 16#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 6#32)) 0#32) = t2_1_0
  generalize (ie_q4_nib w2 17#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 6#32)) 1#32) = t2_1_1
  generalize (ie_q4_nib w2 18#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 6#32)) 2#32) = t2_1_2
  generalize (ie_q4_nib w2 19#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 6#32)) 3#32) = t2_1_3
  generalize (ie_q4_nib w3 0#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 3#32)) 0#32) = t3_0_0
  generalize (ie_q4_nib w3 1#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 3#32)) 1#32) = t3_0_1
  generalize (ie_q4_nib w3 2#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 3#32)) 2#32) = t3_0_2
  generalize (ie_q4_nib w3 3#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 3#32)) 3#32) = t3_0_3
  generalize (ie_q4_nib w3 16#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 7#32)) 0#32) = t3_1_0
  generalize (ie_q4_nib w3 17#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 7#32)) 1#32) = t3_1_1
  generalize (ie_q4_nib w3 18#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 7#32)) 2#32) = t3_1_2
  generalize (ie_q4_nib w3 19#32 - 8#32) * ie_sext8 (ie_byte (aw.load (ab + 7#32)) 3#32) = t3_1_3
  generalize ie_sum_s8x4 (aw.load ab) = s0
  generalize ie_sum_s8x4 (aw.load (ab + 1#32)) = s1
  generalize ie_sum_s8x4 (aw.load (ab + 2#32)) = s2
  generalize ie_sum_s8x4 (aw.load (ab + 3#32)) = s3
  generalize ie_sum_s8x4 (aw.load (ab + 4#32)) = s4
  generalize ie_sum_s8x4 (aw.load (ab + 5#32)) = s5
  generalize ie_sum_s8x4 (aw.load (ab + 6#32)) = s6
  generalize ie_sum_s8x4 (aw.load (ab + 7#32)) = s7
  grind

-- 5. The Q8_0 path

theorem q8_addr : Laws.q8_addr := by
  intro rows nb r b w k
  unfold law_q8_addr
  cases hs : ie_q8_sizes_ok rows nb <;> cases hr : BitVec.ult r rows <;> cases hb : BitVec.ult b nb <;>
    cases hw : BitVec.ult w 8#32 <;> cases hk : BitVec.ult k 32#32 <;>
    simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and, Bool.not_true, Bool.false_or]
  simp only [ie_q8_sizes_ok, Bool.and_eq_true, BitVec.ult_iff_lt] at hs hr hb hw hk
  have ⟨q1, q2⟩ := rows_nb8 rows nb hs.1.1 hs.1.2 hs.2
  have ⟨p1, p2⟩ := r_nb8 rows nb r hs.1.1 hs.1.2 hs.2 hr
  simp only [ie_q8_dst_word, ie_q8_dst_scale, ie_q8_src_qbyte, ie_q8_src_blk, Bool.and_eq_true, BitVec.ult_iff_lt]
  have hb' : b.toNat < nb.toNat := hb
  have hw' : w.toNat < 8 := hw
  have hk' : k.toNat < 32 := hk
  simp only [BitVec.lt_def, BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_ofNat, p1, q1]
  generalize r.toNat * nb.toNat = pn at *; generalize rows.toNat * nb.toNat = qn at *
  simp (disch := omega) only [Nat.mod_eq_of_lt]
  omega

theorem q8_inverse : Laws.q8_inverse := by
  intro rows nb r b w
  unfold law_q8_inverse
  cases hs : ie_q8_sizes_ok rows nb <;> cases hr : BitVec.ult r rows <;> cases hb : BitVec.ult b nb <;>
    cases hw : BitVec.ult w 8#32 <;>
    simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and, Bool.not_true, Bool.false_or]
  simp only [ie_q8_sizes_ok, Bool.and_eq_true, BitVec.ult_iff_lt] at hs hr hb hw
  have ⟨q1, q2⟩ := rows_nb8 rows nb hs.1.1 hs.1.2 hs.2
  have ⟨p1, p2⟩ := r_nb8 rows nb r hs.1.1 hs.1.2 hs.2 hr
  have hb' : b.toNat < nb.toNat := hb
  have hw' : w.toNat < 8 := hw
  have hN : 0 < nb.toNat := by omega
  simp only [ie_q8_word_row, ie_q8_word_blk, ie_q8_word_w, ie_q8_dst_word, Bool.and_eq_true, beq_iff_eq]
  have es : (r * nb + b).toNat = b.toNat + nb.toNat * r.toNat := by
    rw [BitVec.toNat_add, p1, Nat.mod_eq_of_lt (by omega), Nat.mul_comm, Nat.add_comm]
  have e8 : ((r * nb + b) * 8#32 + w).toNat / 8 = b.toNat + nb.toNat * r.toNat := by
    simp only [BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_ofNat, es]
    rw [Nat.mul_comm nb.toNat] at *
    generalize r.toNat * nb.toNat = pn at *
    simp (disch := omega) only [Nat.mod_eq_of_lt]
    omega
  refine ⟨⟨?_, ?_⟩, ?_⟩ <;> apply BitVec.eq_of_toNat_eq
  · rw [BitVec.toNat_udiv, BitVec.toNat_udiv, BitVec.toNat_ofNat, e8, Nat.add_mul_div_left _ _ hN, Nat.div_eq_of_lt hb', Nat.zero_add]
  · rw [BitVec.toNat_umod, BitVec.toNat_udiv, BitVec.toNat_ofNat, e8, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hb']
  · simp only [BitVec.toNat_umod, BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_ofNat, es]
    rw [Nat.mul_comm nb.toNat] at *
    generalize r.toNat * nb.toNat = pn at *
    simp (disch := omega) only [Nat.mod_eq_of_lt]
    omega

theorem q8_onto : Laws.q8_onto := by
  intro rows nb i
  unfold law_q8_onto
  cases hs : ie_q8_sizes_ok rows nb <;> cases h0 : BitVec.ult 0#32 nb <;> cases hi : BitVec.ult i (rows * nb * 8#32) <;>
    simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and, Bool.not_true, Bool.false_or]
  simp only [ie_q8_sizes_ok, Bool.and_eq_true, BitVec.ult_iff_lt] at hs h0 hi
  have ⟨q1, q2⟩ := rows_nb8 rows nb hs.1.1 hs.1.2 hs.2
  have hN : 0 < nb.toNat := h0
  have hN' : nb.toNat < 2048 := hs.1.2
  have hi' : i.toNat < rows.toNat * nb.toNat * 8 := by
    have := BitVec.lt_def.mp hi
    simp only [BitVec.toNat_mul, q1, BitVec.toNat_ofNat] at this
    simp (disch := omega) only [Nat.mod_eq_of_lt] at this; exact this
  simp only [ie_q8_word_row, ie_q8_word_blk, ie_q8_word_w, ie_q8_dst_word, Bool.and_eq_true, beq_iff_eq, BitVec.ult_iff_lt]
  have hq : i.toNat / 8 < rows.toNat * nb.toNat := by omega
  have hd : i.toNat / 8 / nb.toNat < rows.toNat := (Nat.div_lt_iff_lt_mul hN).mpr hq
  have hm : i.toNat / 8 % nb.toNat < nb.toNat := Nat.mod_lt _ hN
  have hdm := Nat.div_add_mod (i.toNat / 8) nb.toNat
  have hle : nb.toNat * (i.toNat / 8 / nb.toNat) ≤ i.toNat / 8 := by omega
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · rw [BitVec.lt_def, BitVec.toNat_udiv, BitVec.toNat_udiv, BitVec.toNat_ofNat]; exact hd
  · rw [BitVec.lt_def, BitVec.toNat_umod, BitVec.toNat_udiv, BitVec.toNat_ofNat]; exact hm
  · apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_udiv, BitVec.toNat_umod, BitVec.toNat_ofNat]
    simp (disch := omega) only [Nat.mod_eq_of_lt (a := 8) (b := 2^32)]
    rw [Nat.mul_comm (i.toNat / 8 / nb.toNat)]
    generalize nb.toNat * (i.toNat / 8 / nb.toNat) = pn at *
    simp (disch := omega) only [Nat.mod_eq_of_lt]
    omega

theorem q8_byte : Laws.q8_byte := by
  intro q0 q1 q2 q3 j; simp only [law_q8_byte, ie_byte, ie_pack4]; bv_decide

theorem q8q8_block : Laws.q8q8_block := by
  intro qw qb aw ab
  simp only [law_q8q8_block, ie_q8q8_block, ie_dot4_ss, ie_q8q8_spec, BitVec.toNat_ofNat, Nat.reduceMod, beq_iff_eq]
  simp only [ie_q8q8_spec.go, ie_q8q8_term, BitVec.reduceULT, BitVec.reduceAdd, BitVec.reduceSub,
    BitVec.reduceAnd, BitVec.reduceHShiftRight, Bool.cond_true, Bool.cond_false, BitVec.add_zero, BitVec.zero_add]
  ac_rfl


-- 6. The Q4_K path (MAT_Q4K)

theorem q4k_nibble : Laws.q4k_nibble := by
  intro l0 l1 l2 l3 h0 h1 h2 h3 b j
  simp only [law_q4k_nibble, ie_q4_nib, ie_pack4, ie_q4k_pack, ie_q4k_src_nib]; bv_decide

theorem q4k_inverse : Laws.q4k_inverse := by
  intro rows nb r b w
  unfold law_q4k_inverse
  cases hs : ie_q4k_sizes_ok rows nb <;> cases hr : BitVec.ult r rows <;> cases hb : BitVec.ult b nb <;>
    cases hw : BitVec.ult w 4#32 <;>
    simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and, Bool.not_true, Bool.false_or]
  simp only [ie_q4k_sizes_ok, Bool.and_eq_true, BitVec.ult_iff_lt, beq_iff_eq] at hs hr hb hw
  have ⟨q1, q2⟩ := rows_nb8 rows nb hs.1.1.1 hs.1.1.2 hs.1.2
  have ⟨p1, p2⟩ := r_nb8 rows nb r hs.1.1.1 hs.1.1.2 hs.1.2 hr
  have hb' : b.toNat < nb.toNat := hb
  have hw' : w.toNat < 4 := hw
  have hN : 0 < nb.toNat := by omega
  simp only [ie_q4_word_row, ie_q4_word_blk, ie_q4_word_w, ie_q4_dst_word, Bool.and_eq_true, beq_iff_eq]
  have es : (r * nb + b).toNat = b.toNat + nb.toNat * r.toNat := by
    rw [BitVec.toNat_add, p1, Nat.mod_eq_of_lt (by omega), Nat.mul_comm, Nat.add_comm]
  have e4 : ((r * nb + b) * 4#32 + w).toNat / 4 = b.toNat + nb.toNat * r.toNat := by
    simp only [BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_ofNat, es]
    rw [Nat.mul_comm nb.toNat] at *
    generalize r.toNat * nb.toNat = pn at *
    simp (disch := omega) only [Nat.mod_eq_of_lt]
    omega
  refine ⟨⟨?_, ?_⟩, ?_⟩ <;> apply BitVec.eq_of_toNat_eq
  · rw [BitVec.toNat_udiv, BitVec.toNat_udiv, BitVec.toNat_ofNat, e4, Nat.add_mul_div_left _ _ hN, Nat.div_eq_of_lt hb', Nat.zero_add]
  · rw [BitVec.toNat_umod, BitVec.toNat_udiv, BitVec.toNat_ofNat, e4, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hb']
  · simp only [BitVec.toNat_umod, BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_ofNat, es]
    rw [Nat.mul_comm nb.toNat] at *
    generalize r.toNat * nb.toNat = pn at *
    simp (disch := omega) only [Nat.mod_eq_of_lt]
    omega

theorem q4k_onto : Laws.q4k_onto := by
  intro rows nb i
  unfold law_q4k_onto
  cases hs : ie_q4k_sizes_ok rows nb <;> cases h0 : BitVec.ult 0#32 nb <;> cases hi : BitVec.ult i (rows * nb * 4#32) <;>
    simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and, Bool.not_true, Bool.false_or]
  simp only [ie_q4k_sizes_ok, Bool.and_eq_true, BitVec.ult_iff_lt, beq_iff_eq] at hs h0 hi
  have ⟨q1, q2⟩ := rows_nb8 rows nb hs.1.1.1 hs.1.1.2 hs.1.2
  have hN : 0 < nb.toNat := h0
  have hi' : i.toNat < rows.toNat * nb.toNat * 4 := by
    have := BitVec.lt_def.mp hi
    simp only [BitVec.toNat_mul, q1, BitVec.toNat_ofNat] at this
    simp (disch := omega) only [Nat.mod_eq_of_lt] at this; exact this
  simp only [ie_q4_word_row, ie_q4_word_blk, ie_q4_word_w, ie_q4_dst_word, Bool.and_eq_true, beq_iff_eq, BitVec.ult_iff_lt]
  have hq : i.toNat / 4 < rows.toNat * nb.toNat := by omega
  have hd : i.toNat / 4 / nb.toNat < rows.toNat := (Nat.div_lt_iff_lt_mul hN).mpr hq
  have hm : i.toNat / 4 % nb.toNat < nb.toNat := Nat.mod_lt _ hN
  have hdm := Nat.div_add_mod (i.toNat / 4) nb.toNat
  have hle : nb.toNat * (i.toNat / 4 / nb.toNat) ≤ i.toNat / 4 := by omega
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · rw [BitVec.lt_def, BitVec.toNat_udiv, BitVec.toNat_udiv, BitVec.toNat_ofNat]; exact hd
  · rw [BitVec.lt_def, BitVec.toNat_umod, BitVec.toNat_udiv, BitVec.toNat_ofNat]; exact hm
  · apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_udiv, BitVec.toNat_umod, BitVec.toNat_ofNat]
    simp (disch := omega) only [Nat.mod_eq_of_lt (a := 4) (b := 2^32)]
    rw [Nat.mul_comm (i.toNat / 4 / nb.toNat)]
    generalize nb.toNat * (i.toNat / 4 / nb.toNat) = pn at *
    simp (disch := omega) only [Nat.mod_eq_of_lt]
    omega

theorem q4k_addr : Laws.q4k_addr := by
  intro rows nb r b w j
  unfold law_q4k_addr
  cases hs : ie_q4k_sizes_ok rows nb <;> cases hr : BitVec.ult r rows <;> cases hb : BitVec.ult b nb <;>
    cases hw : BitVec.ult w 4#32 <;> cases hj : BitVec.ult j 32#32 <;>
    simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and, Bool.not_true, Bool.false_or]
  simp only [ie_q4k_sizes_ok, Bool.and_eq_true, BitVec.ult_iff_lt, beq_iff_eq] at hs hr hb hw hj
  have ⟨q1, q2⟩ := rows_nb8 rows nb hs.1.1.1 hs.1.1.2 hs.1.2
  have ⟨p1, p2⟩ := r_nb8 rows nb r hs.1.1.1 hs.1.1.2 hs.1.2 hr
  have h8 : nb.toNat % 8 = 0 := by
    have := congrArg BitVec.toNat hs.2
    simpa [BitVec.toNat_umod] using this
  have hb' : b.toNat < nb.toNat := hb
  have hw' : w.toNat < 4 := hw
  have hj' : j.toNat < 32 := hj
  have hr' : r.toNat + 1 ≤ rows.toNat := by bv_omega
  -- the products with nb / 8
  have m1 : r.toNat * (nb.toNat / 8) + nb.toNat / 8 ≤ rows.toNat * (nb.toNat / 8) := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hr'
  have m2 : rows.toNat * nb.toNat = 8 * (rows.toNat * (nb.toNat / 8)) := by
    have e : nb.toNat = 8 * (nb.toNat / 8) := by omega
    conv => lhs; rw [e]
    rw [Nat.mul_left_comm]
  have m3 : r.toNat * nb.toNat = 8 * (r.toNat * (nb.toNat / 8)) := by
    have e : nb.toNat = 8 * (nb.toNat / 8) := by omega
    conv => lhs; rw [e]
    rw [Nat.mul_left_comm]
  simp only [ie_q4_dst_word, ie_q4k_sm, ie_q4k_dd, ie_q4k_qs_n, ie_q4k_src_qbyte, ie_q4k_src_blk, Bool.and_eq_true,
    BitVec.ult_iff_lt, BitVec.lt_def, BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_udiv, BitVec.toNat_umod,
    BitVec.toNat_ofNat, p1, q1]
  generalize r.toNat * (nb.toNat / 8) = a at *
  generalize rows.toNat * (nb.toNat / 8) = c at *
  generalize r.toNat * nb.toNat = pn at *
  generalize rows.toNat * nb.toNat = qn at *
  simp (disch := omega) only [Nat.mod_eq_of_lt]
  omega

theorem q4k_dot : Laws.q4k_dot := by
  intro qw qb aw ab
  have hb := q4q8_block qw qb aw ab
  simp only [law_q4q8_block, beq_iff_eq] at hb
  have e : ie_q4k_dot qw qb aw ab = ie_q4q8_block qw qb aw ab (ie_q8_sum aw ab) + 8#32 * ie_q8_sum aw ab := by
    simp only [ie_q4k_dot, ie_q4q8_block, BitVec.sub_add_cancel]
  simp only [law_q4k_dot, beq_iff_eq, e, hb]
  simp only [ie_q4q8_spec, ie_q4k_spec, ie_q8_sum, ie_sum_s8x4, BitVec.toNat_ofNat, Nat.reduceMod]
  simp only [ie_q4q8_spec.go, ie_q4k_spec.go, ie_q4q8_term, ie_q4k_term, ie_q4_word_of, BitVec.reduceULT, BitVec.reduceAdd,
    BitVec.reduceSub, BitVec.reduceAnd, BitVec.reduceMul, BitVec.reduceHShiftRight, Bool.cond_true, Bool.cond_false,
    BitVec.add_zero, BitVec.zero_add]
  generalize Mem.load qw qb = w0
  generalize Mem.load qw (qb + 1#32) = w1
  generalize Mem.load qw (qb + 2#32) = w2
  generalize Mem.load qw (qb + 3#32) = w3
  generalize Mem.load aw ab = x0
  generalize Mem.load aw (ab + 1#32) = x1
  generalize Mem.load aw (ab + 2#32) = x2
  generalize Mem.load aw (ab + 3#32) = x3
  generalize Mem.load aw (ab + 4#32) = x4
  generalize Mem.load aw (ab + 5#32) = x5
  generalize Mem.load aw (ab + 6#32) = x6
  generalize Mem.load aw (ab + 7#32) = x7
  grind



-- 7. The Q6_K path (MAT_Q6K)

theorem q6k_unpack : Laws.q6k_unpack := by
  intro a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 c3 d0 d1 d2 d3 e
  simp only [law_q6k_unpack, ie_q6k_q, ie_q4_nib, ie_pack4, ie_q6k_hpack, ie_q6k_h2]; bv_decide

theorem vw_byte (W H h k i : U32) (hh : h < 2#32) (hk : k < 4#32) (hi : i < 4#32) :
    ie_byte (ie_q6k_vw W H h k) i = ie_q6k_q W H (16#32 * h + 4#32 * k + i) := by
  simp only [ie_byte, ie_q6k_vw, ie_q6k_q, ie_q4_nib]; bv_decide


theorem q6k_dot : Laws.q6k_dot := by
  intro qw qb hw hb aw ab h
  unfold law_q6k_dot
  cases hh : BitVec.ult h 2#32 <;> simp only [Bool.not_false, Bool.true_or, Bool.not_true, Bool.false_or, beq_iff_eq]
  have hh' : h = 0#32 ∨ h = 1#32 := by have := BitVec.ult_iff_lt.mp hh; bv_omega
  rcases hh' with rfl | rfl
  all_goals
    have one : ∀ i : U32, i < 4#32 → ie_sext8 (ie_byte 16843009#32 i) = 1#32 := by
      intro i hi; simp only [ie_sext8, ie_byte]; bv_decide
    simp only [ie_q6k_dot, ie_dot4_us, ie_dot4_ss, ie_q8_hsum, ie_q6k_spec, BitVec.toNat_ofNat, Nat.reduceMod]
    simp (disch := decide) only [one, BitVec.mul_one]
    simp (disch := decide) only [vw_byte, BitVec.reduceMul, BitVec.reduceAdd]
    simp only [ie_q6k_spec.go, ie_q6k_term, ie_q4_word_of, BitVec.reduceULT, BitVec.reduceAdd, BitVec.reduceSub,
      BitVec.reduceAnd, BitVec.reduceMul, BitVec.reduceHShiftRight, Bool.cond_true, Bool.cond_false, BitVec.add_zero,
      BitVec.zero_add, BitVec.zero_mul, BitVec.mul_zero]
    simp only [BitVec.add_assoc, BitVec.reduceAdd]
    generalize Mem.load qw qb = w0
    generalize Mem.load qw (qb + 1#32) = w1
    generalize Mem.load qw (qb + 2#32) = w2
    generalize Mem.load qw (qb + 3#32) = w3
    generalize Mem.load hw hb = hh
    generalize Mem.load aw ab = x0
    generalize Mem.load aw (ab + 1#32) = x1
    generalize Mem.load aw (ab + 2#32) = x2
    generalize Mem.load aw (ab + 3#32) = x3
    generalize Mem.load aw (ab + 4#32) = x4
    generalize Mem.load aw (ab + 5#32) = x5
    generalize Mem.load aw (ab + 6#32) = x6
    generalize Mem.load aw (ab + 7#32) = x7
    grind
theorem q6k_addr : Laws.q6k_addr := by
  intro rows nb r b w h j
  unfold law_q6k_addr
  cases hs : ie_q4k_sizes_ok rows nb <;> cases hr : BitVec.ult r rows <;> cases hb : BitVec.ult b nb <;>
    cases hw : BitVec.ult w 4#32 <;> cases hh : BitVec.ult h 2#32 <;> cases hj : BitVec.ult j 32#32 <;>
    simp only [Bool.false_and, Bool.and_false, Bool.not_false, Bool.true_or, Bool.true_and, Bool.not_true, Bool.false_or]
  simp only [ie_q4k_sizes_ok, Bool.and_eq_true, BitVec.ult_iff_lt, beq_iff_eq] at hs hr hb hw hh hj
  have ⟨q1, q2⟩ := rows_nb8 rows nb hs.1.1.1 hs.1.1.2 hs.1.2
  have ⟨p1, p2⟩ := r_nb8 rows nb r hs.1.1.1 hs.1.1.2 hs.1.2 hr
  have h8 : nb.toNat % 8 = 0 := by
    have := congrArg BitVec.toNat hs.2
    simpa [BitVec.toNat_umod] using this
  have hb' : b.toNat < nb.toNat := hb
  have hw' : w.toNat < 4 := hw
  have hh' : h.toNat < 2 := hh
  have hj' : j.toNat < 32 := hj
  have hr' : r.toNat + 1 ≤ rows.toNat := by bv_omega
  have m1 : r.toNat * (nb.toNat / 8) + nb.toNat / 8 ≤ rows.toNat * (nb.toNat / 8) := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hr'
  have m2 : rows.toNat * nb.toNat = 8 * (rows.toNat * (nb.toNat / 8)) := by
    have e : nb.toNat = 8 * (nb.toNat / 8) := by omega
    conv => lhs; rw [e]
    rw [Nat.mul_left_comm]
  have m3 : r.toNat * nb.toNat = 8 * (r.toNat * (nb.toNat / 8)) := by
    have e : nb.toNat = 8 * (nb.toNat / 8) := by omega
    conv => lhs; rw [e]
    rw [Nat.mul_left_comm]
  simp only [ie_q4_dst_word, ie_q6k_hw, ie_q6k_qw_n, ie_q6k_sc, ie_q6k_d, ie_q6k_qs_n, ie_q6k_src_ql, ie_q6k_src_qh,
    ie_q6k_src_sc, ie_q6k_src_d, ie_q6k_src_blk, Bool.and_eq_true, decide_eq_true_eq,
    BitVec.ult_iff_lt, BitVec.ule_iff_le, BitVec.le_def, BitVec.lt_def, BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_udiv,
    BitVec.toNat_umod, BitVec.toNat_ofNat, p1, q1]
  generalize r.toNat * (nb.toNat / 8) = a at *
  generalize rows.toNat * (nb.toNat / 8) = c at *
  generalize r.toNat * nb.toNat = pn at *
  generalize rows.toNat * nb.toNat = qn at *
  simp (disch := omega) only [Nat.mod_eq_of_lt]

  omega


theorem sw_byte (W H h k i : U32) (hh : h < 2#32) (hk : k < 4#32) (hi : i < 4#32) :
    ie_sext8 (ie_byte (ie_q6k_sw W H h k) i) = ie_q6k_q W H (16#32 * h + 4#32 * k + i) - 32#32 := by
  simp only [ie_sext8, ie_byte, ie_q6k_sw, ie_q6k_vw, ie_q6k_q, ie_q4_nib]; bv_decide

theorem q6k_dots : Laws.q6k_dots := by
  intro qw qb hw hb aw ab h
  unfold law_q6k_dots
  cases hh : BitVec.ult h 2#32 <;> simp only [Bool.not_false, Bool.true_or, Bool.not_true, Bool.false_or, beq_iff_eq]
  have hh' : h = 0#32 ∨ h = 1#32 := by have := BitVec.ult_iff_lt.mp hh; bv_omega
  rcases hh' with rfl | rfl
  all_goals
    simp only [ie_q6k_dots, ie_dot4_ss, ie_q6k_spec, BitVec.toNat_ofNat, Nat.reduceMod]
    simp (disch := decide) only [sw_byte, BitVec.reduceMul, BitVec.reduceAdd]
    simp only [ie_q6k_spec.go, ie_q6k_term, ie_q4_word_of, BitVec.reduceULT, BitVec.reduceAdd, BitVec.reduceSub,
      BitVec.reduceAnd, BitVec.reduceMul, BitVec.reduceHShiftRight, Bool.cond_true, Bool.cond_false, BitVec.add_zero,
      BitVec.zero_add, BitVec.zero_mul, BitVec.mul_zero]
    simp only [BitVec.add_assoc, BitVec.reduceAdd]

theorem q4k_dot_u : Laws.q4k_dot_u := by
  intro qw qb u ub aw ab
  unfold law_q4k_dot_u
  by_cases h : (u.load ub == ie_q4k_lo (qw.load qb) && u.load (ub + 1#32) == ie_q4k_hi (qw.load qb) &&
           u.load (ub + 2#32) == ie_q4k_lo (qw.load (qb + 1#32)) && u.load (ub + 3#32) == ie_q4k_hi (qw.load (qb + 1#32)) &&
           u.load (ub + 4#32) == ie_q4k_lo (qw.load (qb + 2#32)) && u.load (ub + 5#32) == ie_q4k_hi (qw.load (qb + 2#32)) &&
           u.load (ub + 6#32) == ie_q4k_lo (qw.load (qb + 3#32)) && u.load (ub + 7#32) == ie_q4k_hi (qw.load (qb + 3#32))) = true
  · simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨⟨⟨⟨h0, h1⟩, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩ := h
    simp only [ie_q4k_dot_u, ie_q4k_dot, ie_q4q8_word, h0, h1, h2, h3, h4, h5, h6, h7, ie_q4k_lo, ie_q4k_hi,
      Bool.and_self, Bool.not_true, Bool.false_or, beq_self_eq_true]
  · simp only [Bool.not_eq_true] at h
    simp only [h, Bool.not_false, Bool.true_or]

theorem q6k_dots_s : Laws.q6k_dots_s := by
  intro qw qb hw hb sw sb aw ab h
  unfold law_q6k_dots_s
  by_cases hc : (sw.load sb == ie_q6k_sw (qw.load qb) (hw.load hb) h 0#32 &&
           sw.load (sb + 1#32) == ie_q6k_sw (qw.load (qb + 1#32)) (hw.load hb) h 1#32 &&
           sw.load (sb + 2#32) == ie_q6k_sw (qw.load (qb + 2#32)) (hw.load hb) h 2#32 &&
           sw.load (sb + 3#32) == ie_q6k_sw (qw.load (qb + 3#32)) (hw.load hb) h 3#32) = true
  · simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨⟨⟨h0, h1⟩, h2⟩, h3⟩ := hc
    simp only [ie_q6k_dots_s, ie_q6k_dots, h0, h1, h2, h3, Bool.and_self, Bool.not_true, Bool.false_or, beq_self_eq_true]
  · simp only [Bool.not_eq_true] at hc
    simp only [hc, Bool.not_false, Bool.true_or]

end Proof
