#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/translation/stim_translation.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_lines(const std::string& text) {
  std::vector<std::string> lines;
  std::size_t start = 0;
  while (start < text.size()) {
    std::size_t end = text.find('\n', start);
    if (end == std::string::npos) {
      end = text.size();
    }
    if (end > start) {
      lines.push_back(text.substr(start, end - start));
    }
    start = end + 1;
  }
  return lines;
}

bool starts_with(const std::string& s, const char* prefix) {
  return s.rfind(prefix, 0) == 0;
}

std::size_t count_prefix(const std::vector<std::string>& lines, const char* prefix) {
  std::size_t count = 0;
  for (const std::string& line : lines) {
    if (starts_with(line, prefix)) {
      ++count;
    }
  }
  return count;
}

}  // namespace

int main() {
  using namespace qerasure;

  RotatedSurfaceCode code(3);
  const std::size_t qec_rounds = 3;
  const std::size_t extraction_rounds = qec_rounds - 1;
  const std::size_t num_ancillas = code.num_qubits() - code.x_anc_offset();
  const std::size_t num_z_ancillas = code.num_qubits() - code.z_anc_offset();

  const std::string circuit = build_surface_code_stim_circuit(code, qec_rounds);
  if (circuit.empty()) {
    throw std::runtime_error("Stim translation produced an empty circuit");
  }

  const std::vector<std::string> lines = split_lines(circuit);

  if (count_prefix(lines, "H ") != 2 * extraction_rounds) {
    throw std::runtime_error("Unexpected number of H layers in translated circuit");
  }
  // Stim canonically fuses adjacent CX instructions when possible.
  // Without explicit TICKs, one extraction round collapses into one CX instruction.
  if (count_prefix(lines, "CX ") != extraction_rounds) {
    throw std::runtime_error("Unexpected number of CX instructions in translated circuit");
  }
  if (count_prefix(lines, "MR ") != extraction_rounds) {
    throw std::runtime_error("Unexpected number of ancilla measurement rounds");
  }
  if (count_prefix(lines, "M ") != 1) {
    throw std::runtime_error("Expected one final data measurement instruction");
  }
  if (count_prefix(lines, "OBSERVABLE_INCLUDE(0)") != 1) {
    throw std::runtime_error("Expected one logical observable include");
  }

  // Only Z ancillas are exposed as detectors each extraction round, plus final Z-plaquette detectors.
  const std::size_t expected_detectors = extraction_rounds * num_z_ancillas + num_z_ancillas;
  if (count_prefix(lines, "DETECTOR") != expected_detectors) {
    throw std::runtime_error("Unexpected number of DETECTOR instructions");
  }

  bool found_temporal_detector = false;
  for (const std::string& line : lines) {
    if (!starts_with(line, "DETECTOR")) {
      continue;
    }
    std::size_t count = 0;
    std::size_t pos = line.find("rec[");
    while (pos != std::string::npos) {
      ++count;
      pos = line.find("rec[", pos + 1);
    }
    if (count >= 2) {
      found_temporal_detector = true;
      break;
    }
  }
  if (!found_temporal_detector) {
    throw std::runtime_error("Expected temporal detector parity checks between rounds");
  }

  return 0;
}
