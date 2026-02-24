#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/sim/erasure_simulator.h"

namespace qerasure {

enum class PauliError : std::uint8_t {
  NO_ERROR = 0,
  X_ERROR = 1,
  Z_ERROR = 2,
  Y_ERROR = 3,
  DEPOLARIZE = 4
};

struct LoweredErrorParams {
  PauliError error_type = PauliError::NO_ERROR;
  double probability = 0.0;
};

enum class LoweredEventOrigin : std::uint8_t {
  SPREAD = 0,
  RESET = 1,
};

struct LoweredErrorEvent {
  std::size_t qubit_idx;
  PauliError error_type;
  LoweredEventOrigin origin = LoweredEventOrigin::SPREAD;
};

struct LoweringResult {
  // Number of syndrome-extraction rounds inherited from simulator output.
  std::size_t qec_rounds = 0;

  std::vector<std::vector<LoweredErrorEvent>> sparse_cliffords;
  std::vector<std::vector<std::size_t>> clifford_timestep_offsets;
};

// Data-qubit partner ancilla slots used by Stim-like lowering instructions.
enum class PartnerSlot : std::uint8_t {
  X_1 = 0,
  X_2 = 1,
  Z_1 = 2,
  Z_2 = 3,
};

enum class SpreadInstructionType : std::uint8_t {
  X_ERROR = 0,
  Y_ERROR = 1,
  Z_ERROR = 2,
  DEPOLARIZE1 = 3,
  COND_X_ERROR = 4,
  COND_Y_ERROR = 5,
  COND_Z_ERROR = 6,
  ELSE_X_ERROR = 7,
  ELSE_Y_ERROR = 8,
  ELSE_Z_ERROR = 9,
};

struct SpreadTargetOp {
  PauliError error_type = PauliError::NO_ERROR;
  PartnerSlot slot = PartnerSlot::X_1;
};

struct SpreadInstruction {
  SpreadInstructionType type = SpreadInstructionType::X_ERROR;
  double probability = 0.0;
  PartnerSlot target_slot = PartnerSlot::X_1;
};

struct SpreadProgram {
  std::vector<SpreadInstruction> instructions;

  // Stim-like append path. Parses one or many semicolon-separated instructions:
  // e.g. "X_ERROR(0.1) X_1; COND_X_ERROR(0.2) Z_1; ELSE_X_ERROR(1.0) Z_2"
  void append(const std::string& stim_like_program);

  // Strongly-typed append helpers.
  void add_instruction(SpreadInstructionType type, double probability, PartnerSlot target);
  void add_x_error(double probability, PartnerSlot target);
  void add_y_error(double probability, PartnerSlot target);
  void add_z_error(double probability, PartnerSlot target);
  void add_depolarize1(double probability, PartnerSlot target);
  void add_cond_x_error(double probability, PartnerSlot target);
  void add_cond_y_error(double probability, PartnerSlot target);
  void add_cond_z_error(double probability, PartnerSlot target);
  void add_else_x_error(double probability, PartnerSlot target);
  void add_else_y_error(double probability, PartnerSlot target);
  void add_else_z_error(double probability, PartnerSlot target);

  // Backward-compatible legacy helpers.
  void add_error_channel(double probability, std::vector<SpreadTargetOp> targets);
  void add_correlated_error(double probability, std::vector<SpreadTargetOp> targets);
  void add_else_correlated_error(double probability, std::vector<SpreadTargetOp> targets);
};

struct LoweringParams {
  // Legacy fields are retained for compatibility with existing bindings/callers.
  LoweredErrorParams reset_params_;
  std::pair<LoweredErrorParams, LoweredErrorParams> x_ancilla_params_;
  std::pair<LoweredErrorParams, LoweredErrorParams> z_ancilla_params_;

  // New Stim-like lowering standard:
  // - default_data_program applies to all data qubits unless overridden.
  // - per_data_program_overrides[data_idx] applies to specific data qubits.
  SpreadProgram default_data_program;
  std::vector<SpreadProgram> per_data_program_overrides;

  // Preferred constructor for the Stim-like lowering standard:
  // configure reset behavior and default spread program in one object construction.
  explicit LoweringParams(const SpreadProgram& default_program);
  LoweringParams(const SpreadProgram& default_program, const LoweredErrorParams& reset);

  LoweringParams(const LoweredErrorParams& reset, const LoweredErrorParams& ancillas);
  LoweringParams(const LoweredErrorParams& reset, const LoweredErrorParams& x_ancillas,
                const LoweredErrorParams& z_ancillas);
  LoweringParams(const LoweredErrorParams& reset,
                const std::pair<LoweredErrorParams, LoweredErrorParams>& x_ancillas,
                const std::pair<LoweredErrorParams, LoweredErrorParams>& z_ancillas);

  void set_default_data_program(const SpreadProgram& program);
  void set_data_qubit_program(std::size_t data_qubit_idx, const SpreadProgram& program);
};

class Lowerer {
 public:
  explicit Lowerer(const RotatedSurfaceCode& code, const LoweringParams& params);
  LoweringResult lower(const ErasureSimResult& sim_result);

  RotatedSurfaceCode code_;
  LoweringParams params_;
  std::uint64_t rng_state_;

 private:
  struct CompiledTargetOp {
    PauliError error_type = PauliError::NO_ERROR;
    std::size_t qubit_idx = kNoPartner;
  };

  struct CompiledInstruction {
    SpreadInstructionType type = SpreadInstructionType::X_ERROR;
    std::uint64_t threshold = 0;
    CompiledTargetOp target;
  };

  struct CompiledProgram {
    std::vector<CompiledInstruction> instructions;
  };

  std::vector<CompiledProgram> compiled_program_by_data_qubit_;

  void compile_programs();
  static std::uint8_t slot_to_index(PartnerSlot slot);
  std::size_t resolve_data_slot_qubit(std::size_t data_qubit_idx, std::uint8_t slot_index) const;

  std::uint64_t next_random_u64();
  static std::uint64_t probability_to_threshold(double p);
  bool sample_with_threshold(std::uint64_t threshold);
};

}  // namespace qerasure
