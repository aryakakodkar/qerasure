#include "qerasure/core/translation/stim_translation.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qerasure {

namespace {

void append_index_line(std::ostringstream* out, const char* op, const std::vector<std::size_t>& indices) {
  if (indices.empty()) {
    return;
  }
  (*out) << op;
  for (const std::size_t q : indices) {
    (*out) << " " << q;
  }
  (*out) << "\n";
}

void append_cx_step_line(std::ostringstream* out, const std::vector<Gate>& gates, std::size_t step_start,
                         std::size_t step_count) {
  (*out) << "CX";
  for (std::size_t i = 0; i < step_count; ++i) {
    const Gate& gate = gates[step_start + i];
    (*out) << " " << gate.first << " " << gate.second;
  }
  (*out) << "\n";
}

void append_detector_line(std::ostringstream* out, const std::vector<int>& rec_offsets) {
  (*out) << "DETECTOR";
  for (const int offset : rec_offsets) {
    (*out) << " rec[" << offset << "]";
  }
  (*out) << "\n";
}

}  // namespace

std::string build_surface_code_stim_circuit(const RotatedSurfaceCode& code, std::size_t qec_rounds) {
  if (qec_rounds < 2) {
    throw std::invalid_argument("qec_rounds must be >= 2 for Stim circuit generation");
  }

  const std::size_t num_qubits = code.num_qubits();
  const std::size_t num_data = code.x_anc_offset();
  const std::size_t x_anc_offset = code.x_anc_offset();
  const std::size_t z_anc_offset = code.z_anc_offset();
  const std::size_t num_x_anc = z_anc_offset - x_anc_offset;
  const std::size_t num_z_anc = num_qubits - z_anc_offset;
  const std::size_t num_anc = num_x_anc + num_z_anc;
  const std::size_t extraction_rounds = qec_rounds - 1;

  std::vector<std::size_t> data_qubits;
  data_qubits.reserve(num_data);
  for (std::size_t q = 0; q < num_data; ++q) {
    data_qubits.push_back(q);
  }

  std::vector<std::size_t> x_ancillas;
  x_ancillas.reserve(num_x_anc);
  for (std::size_t q = x_anc_offset; q < z_anc_offset; ++q) {
    x_ancillas.push_back(q);
  }

  std::vector<std::size_t> z_ancillas;
  z_ancillas.reserve(num_z_anc);
  for (std::size_t q = z_anc_offset; q < num_qubits; ++q) {
    z_ancillas.push_back(q);
  }

  std::vector<std::size_t> ancillas = x_ancillas;
  ancillas.insert(ancillas.end(), z_ancillas.begin(), z_ancillas.end());

  const std::vector<Gate>& gates = code.gates();
  const std::size_t gates_per_step = code.gates_per_step();

  std::vector<std::vector<std::size_t>> z_ancilla_supports(num_z_anc);
  const std::vector<std::size_t>& partner_map = code.partner_map();
  for (std::size_t zi = 0; zi < num_z_anc; ++zi) {
    const std::size_t z_anc = z_ancillas[zi];
    std::vector<std::size_t>& support = z_ancilla_supports[zi];
    support.reserve(4);
    for (std::size_t step = 0; step < 4; ++step) {
      const std::size_t partner = partner_map[step * num_qubits + z_anc];
      if (partner != kNoPartner && partner < num_data) {
        support.push_back(partner);
      }
    }
    std::sort(support.begin(), support.end());
    support.erase(std::unique(support.begin(), support.end()), support.end());
  }

  std::size_t min_x = std::numeric_limits<std::size_t>::max();
  for (const std::size_t data_q : data_qubits) {
    min_x = std::min(min_x, code.index_to_coord()[data_q].first);
  }

  std::vector<std::size_t> logical_z_data_qubits;
  for (const std::size_t data_q : data_qubits) {
    if (code.index_to_coord()[data_q].first == min_x) {
      logical_z_data_qubits.push_back(data_q);
    }
  }

  std::ostringstream out;

  for (std::size_t round = 0; round < extraction_rounds; ++round) {
    append_index_line(&out, "H", x_ancillas);
    for (std::size_t step = 0; step < 4; ++step) {
      append_cx_step_line(&out, gates, step * gates_per_step, gates_per_step);
    }
    append_index_line(&out, "H", x_ancillas);
    append_index_line(&out, "MR", ancillas);

    if (round == 0) {
      // Initial boundary checks: include only Z-ancilla checks (matches standard memory setup).
      for (std::size_t zi = 0; zi < num_z_anc; ++zi) {
        const std::size_t ancilla_position = num_x_anc + zi;
        const int current_offset = -static_cast<int>(num_anc - ancilla_position);
        append_detector_line(&out, {current_offset});
      }
    } else {
      for (std::size_t ai = 0; ai < num_anc; ++ai) {
        const int current_offset = -static_cast<int>(num_anc - ai);
        const int previous_offset = -static_cast<int>(2 * num_anc - ai);
        append_detector_line(&out, {current_offset, previous_offset});
      }
    }
  }

  append_index_line(&out, "M", data_qubits);

  for (std::size_t zi = 0; zi < num_z_anc; ++zi) {
    std::vector<int> rec_offsets;
    rec_offsets.reserve(1 + z_ancilla_supports[zi].size());

    const std::size_t ancilla_position = num_x_anc + zi;
    const int ancilla_offset_after_data =
        -static_cast<int>(num_data + (num_anc - ancilla_position));
    rec_offsets.push_back(ancilla_offset_after_data);

    for (const std::size_t data_q : z_ancilla_supports[zi]) {
      const int data_offset = -static_cast<int>(num_data - data_q);
      rec_offsets.push_back(data_offset);
    }

    append_detector_line(&out, rec_offsets);
  }

  std::vector<int> logical_rec_offsets;
  logical_rec_offsets.reserve(logical_z_data_qubits.size());
  for (const std::size_t data_q : logical_z_data_qubits) {
    logical_rec_offsets.push_back(-static_cast<int>(num_data - data_q));
  }

  out << "OBSERVABLE_INCLUDE(0)";
  for (const int offset : logical_rec_offsets) {
    out << " rec[" << offset << "]";
  }
  out << "\n";

  return out.str();
}

}  // namespace qerasure
