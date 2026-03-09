#include "surf.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace qerasure::gen {

namespace {

enum class ErasableMode : uint8_t {
  kAll = 0,
  kData = 1,
  kAncilla = 2,
};

std::string uppercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return value;
}

ErasableMode parse_erasable_mode(const std::string& mode) {
  const std::string upper = uppercase(mode);
  if (upper == "ALL") {
    return ErasableMode::kAll;
  }
  if (upper == "DATA") {
    return ErasableMode::kData;
  }
  if (upper == "ANCILLA") {
    return ErasableMode::kAncilla;
  }
  throw std::invalid_argument("erasable_qubits must be one of: ALL, DATA, ANCILLA");
}

bool is_data_qubit(std::size_t q, std::size_t x_anc_offset) {
  return q < x_anc_offset;
}

void append_round_detectors(circuit::ErasureCircuit* circuit, std::size_t num_x_anc,
                            std::size_t num_z_anc, std::size_t num_anc, std::size_t round) {
  for (std::size_t zi = 0; zi < num_z_anc; ++zi) {
    const std::size_t ancilla_position = num_x_anc + zi;
    const uint32_t current_lookback = static_cast<uint32_t>(num_anc - ancilla_position);
    std::vector<uint32_t> rec_lookbacks;
    rec_lookbacks.push_back(current_lookback);
    if (round > 0) {
      const uint32_t previous_lookback = static_cast<uint32_t>(2 * num_anc - ancilla_position);
      rec_lookbacks.push_back(previous_lookback);
    }
    circuit->append_detector(rec_lookbacks);
  }
}

}  // namespace

SurfaceCodeRotated::SurfaceCodeRotated(uint32_t distance) : code_(distance) {}

circuit::ErasureCircuit SurfaceCodeRotated::build_circuit(uint32_t rounds, double erasure_prob,
                                                          std::string erasable_qubits,
                                                          double reset_failure_prob,
                                                          bool ecr_after_each_step,
                                                          bool single_qubit_errors) {
  if (rounds == 0) {
    throw std::invalid_argument("rounds must be > 0");
  }
  if (erasure_prob < 0.0 || erasure_prob > 1.0) {
    throw std::invalid_argument("erasure_prob must be in [0, 1]");
  }
  if (reset_failure_prob < 0.0 || reset_failure_prob > 1.0) {
    throw std::invalid_argument("reset_failure_prob must be in [0, 1]");
  }

  const ErasableMode mode = parse_erasable_mode(erasable_qubits);
  const std::size_t num_qubits = code_.num_qubits();
  const std::size_t num_data = code_.x_anc_offset();
  const std::size_t x_anc_offset = code_.x_anc_offset();
  const std::size_t z_anc_offset = code_.z_anc_offset();
  const std::size_t num_x_anc = z_anc_offset - x_anc_offset;
  const std::size_t num_z_anc = num_qubits - z_anc_offset;
  const std::size_t num_anc = num_x_anc + num_z_anc;
  const std::size_t gates_per_step = code_.gates_per_step();
  const std::vector<Gate>& gates = code_.gates();
  const std::vector<std::size_t>& partner_map = code_.partner_map();
  const auto& coords = code_.index_to_coord();

  std::vector<uint32_t> x_ancillas;
  x_ancillas.reserve(num_x_anc);
  for (std::size_t q = x_anc_offset; q < z_anc_offset; ++q) {
    x_ancillas.push_back(static_cast<uint32_t>(q));
  }

  std::vector<uint32_t> data_qubits;
  data_qubits.reserve(num_data);
  for (std::size_t q = 0; q < num_data; ++q) {
    data_qubits.push_back(static_cast<uint32_t>(q));
  }

  std::vector<uint32_t> all_ancillas;
  all_ancillas.reserve(num_anc);
  for (std::size_t q = x_anc_offset; q < num_qubits; ++q) {
    all_ancillas.push_back(static_cast<uint32_t>(q));
  }

  std::vector<std::vector<std::size_t>> z_ancilla_supports;
  z_ancilla_supports.assign(num_z_anc, {});
  for (std::size_t zi = 0; zi < num_z_anc; ++zi) {
    const std::size_t z_anc = z_anc_offset + zi;
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

  std::vector<uint32_t> logical_observable_lookbacks;
  for (std::size_t q = 0; q < num_data; ++q) {
    if (coords[q].second == 1) {
      logical_observable_lookbacks.push_back(static_cast<uint32_t>(num_data - q));
    }
  }

  std::vector<uint32_t> erasable_targets;
  switch (mode) {
    case ErasableMode::kAll:
      erasable_targets.reserve(num_qubits);
      for (std::size_t q = 0; q < num_qubits; ++q) {
        erasable_targets.push_back(static_cast<uint32_t>(q));
      }
      break;
    case ErasableMode::kData:
      erasable_targets.reserve(x_anc_offset);
      for (std::size_t q = 0; q < x_anc_offset; ++q) {
        erasable_targets.push_back(static_cast<uint32_t>(q));
      }
      break;
    case ErasableMode::kAncilla:
      erasable_targets.reserve(num_qubits - x_anc_offset);
      for (std::size_t q = x_anc_offset; q < num_qubits; ++q) {
        erasable_targets.push_back(static_cast<uint32_t>(q));
      }
      break;
  }

  circuit::ErasureCircuit circuit;

  for (std::size_t round = 0; round < rounds; ++round) {
    circuit.append(circuit::OpCode::H, x_ancillas);
    if (single_qubit_errors && !x_ancillas.empty()) {
      circuit.append(circuit::OpCode::ERASE, x_ancillas, erasure_prob);
    }

    for (std::size_t step = 0; step < 4; ++step) {
      const std::size_t step_start = step * gates_per_step;
      std::vector<uint32_t> cx_targets;
      cx_targets.reserve(gates_per_step * 2);
      std::vector<uint32_t> erase_targets;
      erase_targets.reserve(gates_per_step * 2);

      for (std::size_t i = 0; i < gates_per_step; ++i) {
        const Gate& gate = gates[step_start + i];
        const std::size_t q0 = gate.first;
        const std::size_t q1 = gate.second;
        cx_targets.push_back(static_cast<uint32_t>(q0));
        cx_targets.push_back(static_cast<uint32_t>(q1));

        if (mode == ErasableMode::kAll) {
          erase_targets.push_back(static_cast<uint32_t>(q0));
          erase_targets.push_back(static_cast<uint32_t>(q1));
          continue;
        }

        const bool q0_is_data = is_data_qubit(q0, x_anc_offset);
        const bool q1_is_data = is_data_qubit(q1, x_anc_offset);

        if (mode == ErasableMode::kData) {
          if (q0_is_data && !q1_is_data) {
            erase_targets.push_back(static_cast<uint32_t>(q0));
            erase_targets.push_back(static_cast<uint32_t>(q1));
          } else if (!q0_is_data && q1_is_data) {
            erase_targets.push_back(static_cast<uint32_t>(q1));
            erase_targets.push_back(static_cast<uint32_t>(q0));
          } else {
            throw std::runtime_error("Expected data-ancilla gate pair for DATA directional erasure");
          }
        } else {
          if (!q0_is_data && q1_is_data) {
            erase_targets.push_back(static_cast<uint32_t>(q0));
            erase_targets.push_back(static_cast<uint32_t>(q1));
          } else if (q0_is_data && !q1_is_data) {
            erase_targets.push_back(static_cast<uint32_t>(q1));
            erase_targets.push_back(static_cast<uint32_t>(q0));
          } else {
            throw std::runtime_error(
                "Expected data-ancilla gate pair for ANCILLA directional erasure");
          }
        }
      }

      circuit.append(circuit::OpCode::CX, cx_targets);
      if (mode == ErasableMode::kAll) {
        circuit.append(circuit::OpCode::ERASE2_ANY, erase_targets, erasure_prob);
      } else {
        circuit.append(circuit::OpCode::ERASE2, erase_targets, erasure_prob);
      }
      if (ecr_after_each_step) {
        circuit.append(circuit::OpCode::ECR, erasable_targets, reset_failure_prob);
        if (single_qubit_errors && !erasable_targets.empty()) {
          circuit.append(circuit::OpCode::ERASE, erasable_targets, erasure_prob);
        }
      }
    }

    circuit.append(circuit::OpCode::H, x_ancillas);
    if (single_qubit_errors && !x_ancillas.empty()) {
      circuit.append(circuit::OpCode::ERASE, x_ancillas, erasure_prob);
    }
    if (!ecr_after_each_step) {
      circuit.append(circuit::OpCode::ECR, erasable_targets, reset_failure_prob);
      if (single_qubit_errors && !erasable_targets.empty()) {
        circuit.append(circuit::OpCode::ERASE, erasable_targets, erasure_prob);
      }
    }
    circuit.append(circuit::OpCode::MR, all_ancillas);
    append_round_detectors(&circuit, num_x_anc, num_z_anc, num_anc, round);
  }

  circuit.append(circuit::OpCode::M, data_qubits);

  for (std::size_t zi = 0; zi < num_z_anc; ++zi) {
    const std::size_t ancilla_position = num_x_anc + zi;
    std::vector<uint32_t> rec_lookbacks;
    const uint32_t ancilla_lookback_after_data =
        static_cast<uint32_t>(num_data + (num_anc - ancilla_position));
    rec_lookbacks.push_back(ancilla_lookback_after_data);
    for (const std::size_t data_q : z_ancilla_supports[zi]) {
      rec_lookbacks.push_back(static_cast<uint32_t>(num_data - data_q));
    }
    circuit.append_detector(rec_lookbacks);
  }

  circuit.append_observable_include(logical_observable_lookbacks);

  return circuit;
}

}  // namespace qerasure::gen
