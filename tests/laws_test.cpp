// laws_test.cpp -- run every law of laws/ie_laws.cpp on random inputs (g++).
// The proofs cover all inputs; these tests are a fast first check.
//   g++ -std=c++17 -O2 -o laws_test tests/laws_test.cpp && ./laws_test
#include <cstdio>
#include <random>
#include "../laws/ie_laws.cpp"

static std::mt19937 rng(12345);
static u32 r32() { return rng(); }
static u32 small(u32 m) { return rng() % m; }
// Values near the edges are more useful than uniform ones.
static u32 edge() {
  static const u32 v[] = {0u, 1u, 2u, 7u, 8u, 31u, 32u, 33u, 127u, 128u, 255u, 256u, 0x1FFu, 0x200u,
                          0x3FFFFu, 0x40000u, 0x7FFu, 0x800u, 0x3FFFFFFu, 0x4000000u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu};
  return rng() % 2 ? v[rng() % (sizeof v / sizeof v[0])] : r32();
}

static int fails = 0;
#define CHECK(name, expr)                                    \
  do {                                                       \
    if (!(expr)) {                                           \
      if (fails < 20) std::printf("FAIL %s\n", name);        \
      fails++;                                               \
    }                                                        \
  } while (0)

int main() {
  const int N = 200000;
  for (int it = 0; it < N; it++) {
    u32 rows = it % 2 ? small(0x40001u) : edge(), r = it % 3 ? small(rows + 2u) : edge();
    CHECK("grid_cover", law_grid_cover(rows, r));
    CHECK("grid_unique", law_grid_unique(small(1000), small(9), small(1000), small(9)));
    CHECK("lane_cover", law_lane_cover(edge()));
    CHECK("lane_unique", law_lane_unique(small(33), small(100), small(33), small(100)));
    u32 nb = small(0x201u), b = small(nb + 1u), w = small(5), k = small(17);
    CHECK("q4_addr", law_q4_addr(rows, nb, r, b, w, k));
    CHECK("q4_inverse", law_q4_inverse(rows, nb, r, b, w));
    CHECK("q4_onto", law_q4_onto(rows, nb, small(rows * nb * 4u + 3u)));
    CHECK("act_addr", law_act_addr(nb, b, small(9)));
    CHECK("q4_nibble", law_q4_nibble(r32(), r32(), r32(), r32(), small(34)));
    CHECK("q4_word_of", law_q4_word_of(edge()));
    CHECK("q4q8_word", law_q4q8_word(r32(), r32(), r32()));
    u32 nb8 = it % 4 ? small(0x101u) : small(0x801u), b8 = small(nb8 + 1u);
    CHECK("q8_addr", law_q8_addr(rows, nb8, r, b8, small(9), small(33)));
    CHECK("q8_inverse", law_q8_inverse(rows, nb8, r, b8, small(9)));
    CHECK("q8_onto", law_q8_onto(rows, nb8, small(rows * nb8 * 8u + 3u)));
    CHECK("q8_byte", law_q8_byte(r32(), r32(), r32(), r32(), edge()));
    u32 nbk = it % 2 ? small(0x101u) & ~7u : small(0x801u), bk = small(nbk + 1u);
    CHECK("q4k_addr", law_q4k_addr(rows, nbk, r, bk, small(5), small(34)));
    CHECK("q4k_inverse", law_q4k_inverse(rows, nbk, r, bk, small(5)));
    CHECK("q4k_onto", law_q4k_onto(rows, nbk, small(rows * nbk * 4u + 3u)));
    CHECK("q6k_addr", law_q6k_addr(rows, nbk, r, bk, small(5), small(3), small(34)));
    CHECK("q6k_unpack", law_q6k_unpack(r32(), r32(), r32(), r32(), r32(), r32(), r32(), r32(), r32(), r32(), r32(), r32(), r32(),
                                       r32(), r32(), r32(), small(18)));
    CHECK("q4k_nibble", law_q4k_nibble(r32(), r32(), r32(), r32(), r32(), r32(), r32(), r32(), edge(), small(34)));
  }
  // Memory arrays.
  static u32 xs[4096], qs[64], as[64];
  for (int it = 0; it < 2000; it++) {
    for (u32 &v : xs) v = r32();
    for (u32 &v : qs) v = r32();
    for (u32 &v : as) v = r32();
    Mem x{xs}, q{qs}, a{as};
    CHECK("strided_sum", law_strided_sum(x, small(1500)));
    CHECK("wave_sum", law_wave_sum(x));
    CHECK("q4q8_block", law_q4q8_block(q, small(32), a, small(32)));
    CHECK("q8q8_block", law_q8q8_block(q, small(32), a, small(32)));
    CHECK("q4k_dot", law_q4k_dot(q, small(32), a, small(32)));
    CHECK("q6k_dot", law_q6k_dot(q, small(32), a, small(32), a, small(32), small(3)));
    CHECK("q6k_dots", law_q6k_dots(q, small(32), a, small(32), a, small(32), small(3)));
    {
      static u32 us[16], ss[8];
      const u32 qb = small(32), hb = small(32), h = small(2);
      for (u32 k = 0; k < 4; k++) us[2 * k] = ie_q4k_lo(qs[qb + k]), us[2 * k + 1] = ie_q4k_hi(qs[qb + k]);
      for (u32 k = 0; k < 4; k++) ss[k] = ie_q6k_sw(qs[qb + k], as[hb], h, k);
      Mem u{us}, sw{ss};
      CHECK("q4k_dot_u", law_q4k_dot_u(q, qb, u, 0u, a, small(32)));
      CHECK("q6k_dots_s", law_q6k_dots_s(q, qb, a, hb, sw, 0u, a, small(32), h));
    }
  }
  // Plans: random small plans; the laws must hold when the checker accepts.
  int accepted = 0;
  for (int it = 0; it < 20000; it++) {
    static u32 off[8], size[8], first[8], last[8];
    u32 n = small(9);
    for (u32 i = 0; i < 8; i++) {
      off[i] = small(16) * 256u;
      size[i] = small(1200);
      first[i] = small(6);
      last[i] = first[i] + small(3);
    }
    Plan p{Mem{off}, Mem{size}, Mem{first}, Mem{last}, n, 4096u};
    if (ie_plan_check(p)) accepted++;
    for (int k = 0; k < 50; k++) {
      u32 i = small(9), j = small(9), t = small(9), xb = small(5000);
      CHECK("plan_fits", law_plan_fits(p, i, xb));
      CHECK("plan_aligned", law_plan_aligned(p, i));
      CHECK("plan_no_clobber", law_plan_no_clobber(p, i, j, t, xb));
    }
  }
  std::printf("plans accepted: %d of 20000\n", accepted);
  std::printf(fails ? "laws_test: %d FAIL\n" : "laws_test: all laws pass (%d fails)\n", fails);
  return fails != 0;
}
