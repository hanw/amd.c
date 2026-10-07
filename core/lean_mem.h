// lean_mem.h -- primitives with a fixed meaning in Lean.
// The translator (cpfe-lean) does not translate these bodies. It maps:
//   lean::Mem        -> Mem      (a function from U32 to U32: any array content)
//   lean::load(m, i) -> Mem.load m i
//   lean::Ram        -> Mem      (the same Lean type: memory that can be written)
//   lean::ram_load(r, i)     -> Mem.load r i
//   lean::ram_store(r, i, v) -> Mem.store r i v  (a new memory; r is not changed)
// So a law with a lean::Mem parameter holds for every content of the array.
// C++ reads past the end of the array are undefined behavior; in Lean they
// give some value. Laws must only read inside the array.
#ifndef LEAN_MEM_H
#define LEAN_MEM_H

namespace lean {

// A read-only array of unsigned int.
struct Mem { const unsigned int *p; };

// m[i]
constexpr unsigned int load(Mem m, unsigned int i) { return m.p[i]; }

// A small memory that can be written, as a value (C++ copies it). In C++
// the index must be below RAM_WORDS; in Lean every u32 index is a word.
constexpr unsigned int RAM_WORDS = 1024;
struct Ram { unsigned int v[RAM_WORDS]; };

constexpr unsigned int ram_load(Ram r, unsigned int i) { return r.v[i]; }
constexpr Ram ram_store(Ram r, unsigned int i, unsigned int x) { r.v[i] = x; return r; }

}  // namespace lean

#endif
