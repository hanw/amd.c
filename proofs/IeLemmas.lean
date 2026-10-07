-- Shared lemmas for the proofs of laws/ie_laws.cpp (IeProof.lean).
-- loop_all is adapted from cpp-lean (src/models/mem/lean/MemProof.lean); the
-- Nat-indexed sums sq/ln/lanes/chunk follow the idea of rs/lin_go in cpp-lean
-- (src/lean/TileGen.lean).
import IeLaws
import Std.Tactic.BVDecide
open C
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

namespace Proof

/-- A checker loop "lo < N ? P lo && loop(lo + 1) : true" with enough fuel
gives P at every index in [lo, N). -/
theorem loop_all (N : U32) (P : U32 → Bool) (F : Nat → U32 → U32 → Bool)
    (hF : ∀ f n lo, F (f + 1) n lo = (bif BitVec.ult lo N then (P lo && F f (n - 1#32) (lo + 1#32)) else true)) :
    ∀ (b : Nat) (n lo x : U32), lo ≤ x → x < N → (N - lo).toNat ≤ b → F b n lo = true → P x = true := by
  intro b
  induction b with
  | zero => intro n lo x h1 h2 h3 _; exfalso; bv_omega
  | succ b ih =>
    intro n lo x h1 h2 h3 h
    have hl : BitVec.ult lo N = true := BitVec.ult_iff_lt.mpr (by bv_omega)
    rw [hF] at h
    simp only [hl, Bool.cond_true, Bool.and_eq_true] at h
    by_cases e : lo = x
    · subst e; exact h.1
    · exact ih (n - 1#32) (lo + 1#32) x (by bv_omega) h2 (by bv_omega) h.2

/-- The low nibbles of a Q4 word: byte e of (w & 0x0F0F0F0F) is nibble e. -/
theorem lo_nib (w : U32) (e : U32) (he : e < 4#32) : ie_byte (w &&& 0x0F0F0F0F#32) e = ie_q4_nib w e := by
  simp only [ie_byte, ie_q4_nib]; bv_decide

/-- The high nibbles: byte e of ((w >> 4) & 0x0F0F0F0F) is nibble 16 + e. -/
theorem hi_nib (w : U32) (e : U32) (he : e < 4#32) :
    ie_byte ((w >>> 4) &&& 0x0F0F0F0F#32) e = ie_q4_nib w (16#32 + e) := by
  simp only [ie_byte, ie_q4_nib]; bv_decide

/-- ie_q4_nib only depends on bits 0,1 and 4 of j. -/
theorem nib_eq (w j k : U32) (h : j &&& 19#32 = k &&& 19#32) : ie_q4_nib w j = ie_q4_nib w k := by
  simp only [ie_q4_nib]; bv_decide

-- Plan checker loops

theorem plan_buf (p : Plan) (i : U32) (hc : ie_plan_check p = true) (hi : i < p.n) :
    ie_buf_ok p i = true ∧ ie_pairs_ok p.n p i (i + 1#32) = true := by
  have h := loop_all p.n (fun i => ie_buf_ok p i && ie_pairs_ok p.n p i (i + 1#32))
    (fun f n lo => ie_bufs_ok.go f n p lo) (fun f n lo => by simp only [ie_bufs_ok.go, Bool.and_assoc])
    p.n.toNat p.n 0#32 i (by bv_omega) hi (by bv_omega) hc
  simpa only [Bool.and_eq_true] using h

theorem plan_sep (p : Plan) (i j : U32) (hc : ie_plan_check p = true) (hi : i < p.n) (hj : j < p.n) (hij : i < j) :
    ie_buf_sep p i j = true := by
  have h := (plan_buf p i hc hi).2
  exact loop_all p.n (fun j => ie_buf_sep p i j) (fun f n lo => ie_pairs_ok.go f n p i lo)
    (fun f n lo => by simp only [ie_pairs_ok.go])
    p.n.toNat p.n (i + 1#32) j (by bv_omega) hj (by bv_omega) h


-- Index arithmetic without overflow

theorem mul_nat (x y : U32) (h : x.toNat * y.toNat < 2^32) : (x * y).toNat = x.toNat * y.toNat := by
  rw [BitVec.toNat_mul, Nat.mod_eq_of_lt h]

theorem rows_nb (rows nb : U32) (hR : rows < 262144#32) (hN : nb < 512#32) :
    (rows * nb).toNat = rows.toNat * nb.toNat ∧ rows.toNat * nb.toNat < 2^27 := by
  have hR' : rows.toNat < 2^18 := by bv_omega
  have hN' : nb.toNat < 2^9 := by bv_omega
  have : rows.toNat * nb.toNat < 2^27 :=
    calc rows.toNat * nb.toNat ≤ rows.toNat * 2^9 := Nat.mul_le_mul_left _ (Nat.le_of_lt hN')
      _ < 2^18 * 2^9 := Nat.mul_lt_mul_of_pos_right hR' (by decide)
      _ = 2^27 := by decide
  exact ⟨mul_nat _ _ (by omega), this⟩

theorem r_nb (rows nb r : U32) (hR : rows < 262144#32) (hN : nb < 512#32) (hr : r < rows) :
    (r * nb).toNat = r.toNat * nb.toNat ∧ r.toNat * nb.toNat + nb.toNat ≤ rows.toNat * nb.toNat := by
  have ⟨_, h2⟩ := rows_nb rows nb hR hN
  have hr' : r.toNat + 1 ≤ rows.toNat := by bv_omega
  have : r.toNat * nb.toNat + nb.toNat ≤ rows.toNat * nb.toNat := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hr'
  exact ⟨mul_nat _ _ (by omega), this⟩


theorem rows_nb8 (rows nb : U32) (hR : rows < 262144#32) (hN : nb < 2048#32) (hP : rows * nb < 67108864#32) :
    (rows * nb).toNat = rows.toNat * nb.toNat ∧ rows.toNat * nb.toNat < 2^26 := by
  have hR' : rows.toNat < 2^18 := by bv_omega
  have hN' : nb.toNat < 2^11 := by bv_omega
  have h29 : rows.toNat * nb.toNat < 2^29 :=
    calc rows.toNat * nb.toNat ≤ rows.toNat * 2^11 := Nat.mul_le_mul_left _ (Nat.le_of_lt hN')
      _ < 2^18 * 2^11 := Nat.mul_lt_mul_of_pos_right hR' (by decide)
      _ = 2^29 := by decide
  have e := mul_nat rows nb (by omega)
  refine ⟨e, ?_⟩
  have := BitVec.lt_def.mp hP
  rw [e] at this
  simpa using this

theorem r_nb8 (rows nb r : U32) (hR : rows < 262144#32) (hN : nb < 2048#32) (hP : rows * nb < 67108864#32)
    (hr : r < rows) :
    (r * nb).toNat = r.toNat * nb.toNat ∧ r.toNat * nb.toNat + nb.toNat ≤ rows.toNat * nb.toNat := by
  have ⟨_, h2⟩ := rows_nb8 rows nb hR hN hP
  have hr' : r.toNat + 1 ≤ rows.toNat := by bv_omega
  have : r.toNat * nb.toNat + nb.toNat ≤ rows.toNat * nb.toNat := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hr'
  exact ⟨mul_nat _ _ (by omega), this⟩

-- Strided and sequential sums

/-- f a + f (a+1) + ... + f (N-1). -/
def sq (f : Nat → U32) (N a : Nat) : U32 := if a < N then f a + sq f N (a + 1) else 0#32
termination_by N - a

/-- One lane: f a + f (a+32) + ... while < N. -/
def ln (f : Nat → U32) (N a : Nat) : U32 := if a < N then f a + ln f N (a + 32) else 0#32
termination_by N - a

/-- Lanes a, a+1, ..., a+m-1. -/
def lanes (f : Nat → U32) (N a : Nat) : Nat → U32
  | 0 => 0#32
  | m + 1 => ln f N a + lanes f N (a + 1) m

/-- Guarded chunk: Σ_{l<m} [a+l<N] f (a+l). -/
def chunk (f : Nat → U32) (N a : Nat) : Nat → U32
  | 0 => 0#32
  | m + 1 => (if a < N then f a else 0#32) + chunk f N (a + 1) m

theorem sq_ge (f : Nat → U32) (N a : Nat) (h : N ≤ a) : sq f N a = 0#32 := by
  rw [sq]; simp [show ¬ a < N by omega]

theorem ln_ge (f : Nat → U32) (N a : Nat) (h : N ≤ a) : ln f N a = 0#32 := by
  rw [ln]; simp [show ¬ a < N by omega]

theorem chunk_ge (f : Nat → U32) (N : Nat) (m : Nat) : ∀ a, N ≤ a → chunk f N a m = 0#32 := by
  induction m with
  | zero => intro a _; rfl
  | succ m ih => intro a h; simp [chunk, show ¬ a < N by omega, ih (a + 1) (by omega)]

theorem lanes_ge (f : Nat → U32) (N : Nat) (m : Nat) : ∀ a, N ≤ a → lanes f N a m = 0#32 := by
  induction m with
  | zero => intro a _; rfl
  | succ m ih => intro a h; simp [lanes, ln_ge f N a h, ih (a + 1) (by omega)]

theorem sq_chunk (f : Nat → U32) (N : Nat) (m : Nat) : ∀ a, sq f N a = chunk f N a m + sq f N (a + m) := by
  induction m with
  | zero => intro a; simp [chunk]
  | succ m ih =>
    intro a
    by_cases h : a < N
    · rw [sq, if_pos h, ih (a + 1), chunk, if_pos h, BitVec.add_assoc, Nat.add_assoc, Nat.add_comm 1 m]
    · rw [sq_ge f N a (by omega), chunk_ge f N (m + 1) a (by omega), sq_ge f N (a + (m + 1)) (by omega)]; rfl

theorem ln_step (f : Nat → U32) (N a : Nat) : ln f N a = (if a < N then f a else 0#32) + ln f N (a + 32) := by
  by_cases h : a < N
  · rw [ln, if_pos h, if_pos h]
  · rw [ln_ge f N a (by omega), if_neg h, ln_ge f N (a + 32) (by omega)]; rfl

theorem lanes_chunk (f : Nat → U32) (N : Nat) (m : Nat) : ∀ a, lanes f N a m = chunk f N a m + lanes f N (a + 32) m := by
  induction m with
  | zero => intro a; rfl
  | succ m ih =>
    intro a
    simp only [lanes, chunk]
    rw [ln_step f N a, ih (a + 1), show a + 1 + 32 = a + 32 + 1 by omega]
    generalize (if a < N then f a else 0#32) = u
    ac_rfl

theorem lanes_sq (f : Nat → U32) (N : Nat) : ∀ k a, N - a ≤ 32 * k → lanes f N a 32 = sq f N a := by
  intro k
  induction k with
  | zero => intro a hk; rw [lanes_ge f N 32 a (by omega), sq_ge f N a (by omega)]
  | succ k ih =>
    intro a hk
    rw [lanes_chunk, ih (a + 32) (by omega), sq_chunk f N 32 a]

/-- The generated loops in terms of the Nat sums. -/
theorem seq_go (x : Mem) (nb : U32) : ∀ (fuel : Nat) (n lo : U32), nb.toNat - lo.toNat ≤ fuel →
    ie_seq_sum.go fuel n x lo nb = sq (fun i => x.load (BitVec.ofNat 32 i)) nb.toNat lo.toNat := by
  intro fuel
  induction fuel with
  | zero => intro n lo h; rw [ie_seq_sum.go, sq_ge _ _ _ (by omega)]
  | succ k ih =>
    intro n lo h
    rw [ie_seq_sum.go]
    by_cases hl : lo < nb
    · have hl' : lo.toNat < nb.toNat := hl
      have e : (lo + 1#32).toNat = lo.toNat + 1 := by bv_omega
      rw [show BitVec.ult lo nb = true from BitVec.ult_iff_lt.mpr hl, Bool.cond_true, sq, if_pos hl',
        ih _ _ (by omega), e, BitVec.ofNat_toNat, BitVec.setWidth_eq]
    · have hl' : ¬ lo.toNat < nb.toNat := hl
      rw [show BitVec.ult lo nb = false from Bool.eq_false_iff.mpr (fun h' => hl (BitVec.ult_iff_lt.mp h')), Bool.cond_false, sq_ge _ _ _ (by omega)]

theorem lane_go (x : Mem) (nb : U32) (hN : nb.toNat < 2^30) : ∀ (fuel : Nat) (n b : U32), nb.toNat - b.toNat ≤ fuel →
    ie_lane_sum.go fuel n x b nb = ln (fun i => x.load (BitVec.ofNat 32 i)) nb.toNat b.toNat := by
  intro fuel
  induction fuel with
  | zero => intro n b h; rw [ie_lane_sum.go, ln_ge _ _ _ (by omega)]
  | succ k ih =>
    intro n b h
    rw [ie_lane_sum.go]
    by_cases hl : b < nb
    · have hl' : b.toNat < nb.toNat := hl
      have e : (b + 32#32).toNat = b.toNat + 32 := by bv_omega
      rw [show BitVec.ult b nb = true from BitVec.ult_iff_lt.mpr hl, Bool.cond_true, ln, if_pos hl',
        ih _ _ (by omega), e, BitVec.ofNat_toNat, BitVec.setWidth_eq]
    · have hl' : ¬ b.toNat < nb.toNat := hl
      rw [show BitVec.ult b nb = false from Bool.eq_false_iff.mpr (fun h' => hl (BitVec.ult_iff_lt.mp h')), Bool.cond_false, ln_ge _ _ _ (by omega)]

theorem lanes_go (x : Mem) (nb : U32) (hN : nb.toNat < 2^30) : ∀ (fuel : Nat) (n l : U32), l.toNat ≤ 32 → 32 - l.toNat ≤ fuel →
    ie_lanes_sum.go fuel n x l nb = lanes (fun i => x.load (BitVec.ofNat 32 i)) nb.toNat l.toNat (32 - l.toNat) := by
  intro fuel
  induction fuel with
  | zero => intro n l h1 h2; rw [ie_lanes_sum.go, show 32 - l.toNat = 0 by omega]; rfl
  | succ k ih =>
    intro n l h1 h2
    rw [ie_lanes_sum.go]
    by_cases hl : l < 32#32
    · have hl' : l.toNat < 32 := hl
      have e : (l + 1#32).toNat = l.toNat + 1 := by bv_omega
      rw [show BitVec.ult l 32#32 = true from BitVec.ult_iff_lt.mpr hl, Bool.cond_true,
        ih _ _ (by omega) (by omega), e, show 32 - l.toNat = (32 - (l.toNat + 1)) + 1 by omega, lanes,
        ie_lane_sum, lane_go x nb hN _ _ _ (by omega)]
    · have hl' : ¬ l.toNat < 32 := hl
      rw [show BitVec.ult l 32#32 = false from Bool.eq_false_iff.mpr (fun h' => hl (BitVec.ult_iff_lt.mp h')), Bool.cond_false,
        show 32 - l.toNat = 0 by omega]; rfl


end Proof
