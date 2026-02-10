#pragma once

#include <cstdint>

namespace qerasure::internal {

inline std::uint64_t splitmix64_next(std::uint64_t* state) {
  std::uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

}  // namespace qerasure::internal
