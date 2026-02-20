#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/translation/stim_translation.h"
#include "stim/circuit/circuit.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

std::size_t parse_or_default(char* arg, std::size_t fallback) {
  if (arg == nullptr) {
    return fallback;
  }
  return static_cast<std::size_t>(std::stoull(arg));
}

bool parse_string_mode(char* arg) {
  if (arg == nullptr) {
    return false;
  }
  const std::string mode(arg);
  return mode == "string" || mode == "--string";
}

}  // namespace

int main(int argc, char* argv[]) {
  const std::size_t circuits = parse_or_default(argc > 1 ? argv[1] : nullptr, 10000);
  const std::size_t distance = parse_or_default(argc > 2 ? argv[2] : nullptr, 15);
  const std::size_t qec_rounds = parse_or_default(argc > 3 ? argv[3] : nullptr, 16);
  const bool include_string_serialization = parse_string_mode(argc > 4 ? argv[4] : nullptr);

  qerasure::RotatedSurfaceCode code(distance);
  std::size_t sink = 0;

  const auto start = std::chrono::steady_clock::now();
  if (include_string_serialization) {
    for (std::size_t i = 0; i < circuits; ++i) {
      const std::string text = qerasure::build_surface_code_stim_circuit(code, qec_rounds);
      sink ^= text.size();
    }
  } else {
    for (std::size_t i = 0; i < circuits; ++i) {
      stim::Circuit circuit = qerasure::build_surface_code_stim_circuit_object(code, qec_rounds);
      sink ^= circuit.operations.size();
    }
  }
  const auto end = std::chrono::steady_clock::now();

  const std::chrono::duration<double> elapsed = end - start;
  std::cout << "ElapsedSeconds: " << elapsed.count() << "\n";
  std::cout << "Circuits/sec: " << (static_cast<double>(circuits) / elapsed.count()) << "\n";
  std::cout << "Mode: " << (include_string_serialization ? "build+string" : "build-only") << "\n";
  std::cout << "Sink: " << sink << "\n";
  return 0;
}
