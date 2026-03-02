#pragma once

#include <array>
#include <cstdint>

namespace qerasure::simulator {

class FastRng {
 public:
  explicit FastRng(std::uint64_t seed, std::uint64_t stream_id = 0) {
    std::uint64_t state = seed ^ (stream_id * 0x9E3779B97F4A7C15ULL);
    state_[0] = splitmix64_next(&state);
    state_[1] = splitmix64_next(&state);
    state_[2] = splitmix64_next(&state);
    state_[3] = splitmix64_next(&state);
  }

  std::uint64_t next_u64() {
    const std::uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
    const std::uint64_t t = state_[1] << 17;

    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];

    state_[2] ^= t;
    state_[3] = rotl(state_[3], 45);
    return result;
  }

 private:
  static std::uint64_t splitmix64_next(std::uint64_t* state) {
    std::uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  static std::uint64_t rotl(std::uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  std::array<std::uint64_t, 4> state_{};
};

}  // namespace qerasure::simulator
