#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/lowering/lowering.h"
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

std::size_t first_index_of_prefix(const std::vector<std::string>& lines, const char* prefix) {
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (starts_with(lines[i], prefix)) {
      return i;
    }
  }
  return static_cast<std::size_t>(-1);
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

  LoweringResult lowering;
  lowering.qec_rounds = 2;
  lowering.sparse_cliffords.resize(1);
  lowering.clifford_timestep_offsets.resize(1);
  // Timesteps 0..8 plus terminal offset.
  lowering.clifford_timestep_offsets[0] = {0, 1, 1, 1, 1, 2, 2, 2, 2, 3};
  lowering.sparse_cliffords[0].push_back({6, PauliError::Z_ERROR, LoweredEventOrigin::SPREAD});  // t=0
  lowering.sparse_cliffords[0].push_back({5, PauliError::X_ERROR, LoweredEventOrigin::RESET});   // t=4
  lowering.sparse_cliffords[0].push_back({1, PauliError::Y_ERROR, LoweredEventOrigin::SPREAD});  // t=8

  const std::string lowered_circuit =
      build_logically_equivalent_erasure_stim_circuit(code, lowering, 0);
  const std::vector<std::string> lowered_lines = split_lines(lowered_circuit);

  if (count_prefix(lowered_lines, "X_ERROR(1)") != 1) {
    throw std::runtime_error("Expected one injected X_ERROR(1) from lowering events");
  }
  if (count_prefix(lowered_lines, "Z_ERROR(1)") != 1) {
    throw std::runtime_error("Expected one injected Z_ERROR(1) from lowering events");
  }
  if (count_prefix(lowered_lines, "Y_ERROR(1)") != 1) {
    throw std::runtime_error("Expected one injected Y_ERROR(1) from lowering events");
  }
  const std::size_t first_cx = first_index_of_prefix(lowered_lines, "CX ");
  const std::size_t first_x_error = first_index_of_prefix(lowered_lines, "X_ERROR(1)");
  const std::size_t first_z_error = first_index_of_prefix(lowered_lines, "Z_ERROR(1)");
  std::size_t first_round_second_h = static_cast<std::size_t>(-1);
  std::size_t first_round_mr = static_cast<std::size_t>(-1);
  std::size_t h_seen = 0;
  std::size_t mr_seen = 0;
  for (std::size_t i = 0; i < lowered_lines.size(); ++i) {
    if (starts_with(lowered_lines[i], "H ")) {
      ++h_seen;
      if (h_seen == 2) {
        first_round_second_h = i;
      }
    }
    if (starts_with(lowered_lines[i], "MR ")) {
      ++mr_seen;
      if (mr_seen == 1) {
        first_round_mr = i;
      }
    }
  }
  if (first_cx == static_cast<std::size_t>(-1) || first_x_error == static_cast<std::size_t>(-1) ||
      first_round_second_h == static_cast<std::size_t>(-1) || first_round_mr == static_cast<std::size_t>(-1) ||
      first_x_error <= first_round_second_h || first_x_error >= first_round_mr) {
    throw std::runtime_error(
        "Reset-origin lowering errors must be injected after second H and before MR in the same round");
  }
  if (first_z_error == static_cast<std::size_t>(-1) || first_z_error <= first_cx) {
    throw std::runtime_error("Spread-origin lowering errors must be injected after the corresponding CX step");
  }

  // Virtual decoder circuit: correlated-flag carry-over must suppress ELSE instructions even when
  // the correlated instruction has no emitted target error.
  SpreadProgram virtual_program;
  virtual_program.add_correlated_error(1.0, {{PauliError::NO_ERROR, PartnerSlot::X_2}});
  virtual_program.add_else_correlated_error(1.0, {{PauliError::X_ERROR, PartnerSlot::X_1}});
  const std::string virtual_circuit =
      build_virtual_decoder_stim_circuit(code, 1, virtual_program, 0.25, true);
  const std::vector<std::string> virtual_lines = split_lines(virtual_circuit);
  if (count_prefix(virtual_lines, "E(") == 0 || count_prefix(virtual_lines, "ELSE_CORRELATED_ERROR(") == 0) {
    throw std::runtime_error(
        "Virtual decoder translation must preserve correlated/else-correlated instructions");
  }
  if (count_prefix(virtual_lines, "X_ERROR(") != 0) {
    throw std::runtime_error(
        "ELSE_CORRELATED_ERROR should be suppressed when correlated branch fires with probability 1");
  }

  return 0;
}
