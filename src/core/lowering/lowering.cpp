#include "qerasure/core/lowering/lowering.h"

#include <limits>
#include <utility>
#include <vector>

#include "../sim/internal/fast_rng.h"

namespace qerasure {

Lowerer::Lowerer(const RotatedSurfaceCode& code, const LoweringParams& params)
    : code_(code), params_(params), rng_state_(0xA24BAED4963EE407ULL) {}

std::uint64_t Lowerer::next_random_u64() {
    return internal::splitmix64_next(&rng_state_);
}

std::uint64_t Lowerer::probability_to_threshold(double p) {
    if (p <= 0.0) {
        return 0;
    }
    if (p >= 1.0) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(
        p * static_cast<long double>(std::numeric_limits<std::uint64_t>::max()));
}

bool Lowerer::sample_with_threshold(std::uint64_t threshold) {
    if (threshold == 0) {
        return false;
    }
    if (threshold == std::numeric_limits<std::uint64_t>::max()) {
        return true;
    }
    return next_random_u64() <= threshold;
}

LoweringParams::LoweringParams(const LoweredErrorParams& reset, const LoweredErrorParams& ancillas) {
    reset_params_ = reset;
    x_ancilla_params_ = {ancillas, ancillas};
    z_ancilla_params_ = {ancillas, ancillas};
}

LoweringParams::LoweringParams(const LoweredErrorParams& reset, const LoweredErrorParams& x_ancillas,
                    const LoweredErrorParams& z_ancillas) {
    reset_params_ = reset;
    x_ancilla_params_ = {x_ancillas, x_ancillas};
    z_ancilla_params_ = {z_ancillas, z_ancillas};
}

LoweringParams::LoweringParams(const LoweredErrorParams& reset, const std::pair<LoweredErrorParams, LoweredErrorParams>& x_ancillas,
                    const std::pair<LoweredErrorParams, LoweredErrorParams>& z_ancillas) {
    reset_params_ = reset;
    x_ancilla_params_ = x_ancillas;
    z_ancilla_params_ = z_ancillas;
}

LoweringResult Lowerer::lower(const ErasureSimResult& sim_result) {
    LoweringResult result;
    
    // Precompute partner ancillas
    std::vector<std::pair<std::size_t, std::size_t>> x_ancilla_partners_(code_.x_anc_offset(), {kNoPartner, kNoPartner});
    std::vector<std::pair<std::size_t, std::size_t>> z_ancilla_partners_(code_.x_anc_offset(), {kNoPartner, kNoPartner});

    for (std::size_t data_qubit_idx = 0; data_qubit_idx < code_.x_anc_offset(); ++data_qubit_idx) {
        // Update x_ancilla_partners_ and z_ancilla_partners_
        for (std::size_t step = 0; step < 4; ++step) {
            std::size_t partner = code_.partner_map()[step * code_.num_qubits() + data_qubit_idx];
            if (partner != kNoPartner) {
                if (partner >= code_.x_anc_offset() && partner < code_.z_anc_offset()) {
                    if (x_ancilla_partners_[data_qubit_idx].first == kNoPartner) {
                        x_ancilla_partners_[data_qubit_idx].first = partner;
                    } else {
                        x_ancilla_partners_[data_qubit_idx].second = partner;
                    }
                } else if (partner >= code_.z_anc_offset()) {
                    if (z_ancilla_partners_[data_qubit_idx].first == kNoPartner) {
                        z_ancilla_partners_[data_qubit_idx].first = partner;
                    } else {
                        z_ancilla_partners_[data_qubit_idx].second = partner;
                    }
                }
            }
        }
    }

    const std::size_t num_qubits = code_.num_qubits();
    const auto& partner_map = code_.partner_map();
    const std::size_t x_anc_offset = code_.x_anc_offset();
    const std::size_t z_anc_offset = code_.z_anc_offset();

    struct StepSpreadInfo {
        std::size_t partner = kNoPartner;
        PauliError error_type = PauliError::NO_ERROR;
        std::uint64_t threshold = 0;
    };

    // Precompute gate-spread behavior for each [step][erased_qubit] entry.
    std::vector<StepSpreadInfo> spread_info(4 * num_qubits);
    const std::uint64_t x_first_threshold = probability_to_threshold(params_.x_ancilla_params_.first.probability);
    const std::uint64_t x_second_threshold = probability_to_threshold(params_.x_ancilla_params_.second.probability);
    const std::uint64_t z_first_threshold = probability_to_threshold(params_.z_ancilla_params_.first.probability);
    const std::uint64_t z_second_threshold = probability_to_threshold(params_.z_ancilla_params_.second.probability);

    for (std::size_t step = 0; step < 4; ++step) {
        const std::size_t base = step * num_qubits;
        for (std::size_t erased_qubit = 0; erased_qubit < num_qubits; ++erased_qubit) {
            const std::size_t partner = partner_map[base + erased_qubit];
            if (partner == kNoPartner) {
                continue;
            }

            StepSpreadInfo info;
            info.partner = partner;

            if (erased_qubit < x_anc_offset) {
                if (partner == x_ancilla_partners_[erased_qubit].first) {
                    info.error_type = params_.x_ancilla_params_.first.error_type;
                    info.threshold = x_first_threshold;
                } else if (partner == x_ancilla_partners_[erased_qubit].second) {
                    info.error_type = params_.x_ancilla_params_.second.error_type;
                    info.threshold = x_second_threshold;
                } else if (partner == z_ancilla_partners_[erased_qubit].first) {
                    info.error_type = params_.z_ancilla_params_.first.error_type;
                    info.threshold = z_first_threshold;
                } else if (partner == z_ancilla_partners_[erased_qubit].second) {
                    info.error_type = params_.z_ancilla_params_.second.error_type;
                    info.threshold = z_second_threshold;
                }
            } else if (partner < x_anc_offset) {
                // Ancilla erased -> data partner. Ancilla erasure to data qubit spread not currently supported.
                const std::size_t data_qubit = partner;
                if (erased_qubit < z_anc_offset) {
                    if (erased_qubit == x_ancilla_partners_[data_qubit].first) {
                        info.error_type = params_.x_ancilla_params_.first.error_type;
                        info.threshold = x_first_threshold;
                    } else if (erased_qubit == x_ancilla_partners_[data_qubit].second) {
                        info.error_type = params_.x_ancilla_params_.second.error_type;
                        info.threshold = x_second_threshold;
                    }
                } else {
                    if (erased_qubit == z_ancilla_partners_[data_qubit].first) {
                        info.error_type = params_.z_ancilla_params_.first.error_type;
                        info.threshold = z_first_threshold;
                    } else if (erased_qubit == z_ancilla_partners_[data_qubit].second) {
                        info.error_type = params_.z_ancilla_params_.second.error_type;
                        info.threshold = z_second_threshold;
                    }
                }
            }

            spread_info[base + erased_qubit] = info;
        }
    }

    result.sparse_cliffords.resize(sim_result.sparse_erasures.size());
    result.clifford_timestep_offsets.resize(sim_result.erasure_timestep_offsets.size());
    const std::uint64_t reset_threshold = probability_to_threshold(params_.reset_params_.probability);
    // Reuse state buffers across shots to avoid repeated allocation/initialization.
    std::vector<std::uint8_t> erased_state(num_qubits, 0); // 0 = not erased, 1 = erased
    std::vector<std::size_t> erased_pos(num_qubits, kNoPartner);
    std::vector<std::size_t> erased_qubits;
    erased_qubits.reserve(num_qubits / 4 + 1);

    for (std::size_t shot = 0; shot < sim_result.sparse_erasures.size(); ++shot) {
        std::size_t event_index = 0;
        std::size_t num_lowering_events = 0;

        const auto& events = sim_result.sparse_erasures[shot];
        const auto& offsets = sim_result.erasure_timestep_offsets[shot];
        auto& lowered_events = result.sparse_cliffords[shot];
        auto& lowered_offsets = result.clifford_timestep_offsets[shot];
        lowered_offsets.assign(offsets.size(), 0);
        // Reserve near event count to avoid repeated growth in high-noise settings.
        lowered_events.clear();
        lowered_events.reserve(events.size() + events.size() / 2 + 8);

        erased_qubits.clear();

        for (std::size_t t = 0; t + 1 < offsets.size(); ++t) {
            const std::size_t end_index = offsets[t + 1];

            // Apply events at timestep t to update persistent erasure state.
            for (; event_index < end_index; ++event_index) {
                const EventType event_type = events[event_index].event_type;
                const std::size_t qubit_idx = events[event_index].qubit_idx;

                if (event_type == EventType::ERASURE) {
                    if (erased_state[qubit_idx] == 0) {
                        erased_state[qubit_idx] = 1;
                        erased_pos[qubit_idx] = erased_qubits.size();
                        erased_qubits.push_back(qubit_idx);
                    }
                } else if (event_type == EventType::RESET) {
                    if (erased_state[qubit_idx] != 0) {
                        erased_state[qubit_idx] = 0;
                        const std::size_t pos = erased_pos[qubit_idx];
                        const std::size_t last = erased_qubits.back();
                        erased_qubits[pos] = last;
                        erased_pos[last] = pos;
                        erased_qubits.pop_back();
                        erased_pos[qubit_idx] = kNoPartner;
                    }
                    if (params_.reset_params_.error_type != PauliError::NO_ERROR &&
                        sample_with_threshold(reset_threshold)) {
                        lowered_events.push_back({qubit_idx, params_.reset_params_.error_type});
                        ++num_lowering_events;
                    }
                }
            }

            // Only gate timesteps (4 per round) have partner interactions.
            if (t < offsets.size() - 2) {
                const std::size_t step = t % 4;
                const std::size_t base = step * num_qubits;

                // Hot path: iterate only currently erased qubits.
                for (const std::size_t erased_qubit : erased_qubits) {
                    const StepSpreadInfo& info = spread_info[base + erased_qubit];
                    if (info.partner == kNoPartner || info.error_type == PauliError::NO_ERROR ||
                        !sample_with_threshold(info.threshold)) {
                        continue;
                    }
                    lowered_events.push_back({info.partner, info.error_type});
                    ++num_lowering_events;
                }
            }

            lowered_offsets[t + 1] = num_lowering_events;
        }

        // Clear persistent erasure state for the next shot.
        for (const std::size_t q : erased_qubits) {
            erased_state[q] = 0;
            erased_pos[q] = kNoPartner;
        }
    }

    return result;
}

} // namespace qerasure
