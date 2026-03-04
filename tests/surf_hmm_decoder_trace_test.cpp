#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "core/circuit/compile.h"
#include "core/circuit/erasure_model.h"
#include "core/decode/surf_hmm_decoder.h"
#include "core/gen/surf.h"
#include "core/model/pauli_channel.h"
#include "core/simulator/erasure_sampler.h"
#include "core/simulator/injector.h"

namespace {

uint8_t check_outcome_to_bit(qerasure::simulator::CheckOutcome outcome) {
	switch (outcome) {
		case qerasure::simulator::CheckOutcome::TruePositive:
		case qerasure::simulator::CheckOutcome::FalsePositive:
			return 1;
		case qerasure::simulator::CheckOutcome::FalseNegative:
		case qerasure::simulator::CheckOutcome::TrueNegative:
			return 0;
	}
	return 0;
}

std::vector<uint8_t> extract_check_bits(const qerasure::simulator::SampledShot& shot) {
	std::vector<uint8_t> bits;
	for (const auto& group : shot.operation_groups) {
		for (const auto& check : group.checks) {
			bits.push_back(check_outcome_to_bit(check.outcome));
		}
	}
	return bits;
}

}  // namespace

int main() {
	using namespace qerasure::circuit;    // NOLINT
	using namespace qerasure::decode;     // NOLINT
	using namespace qerasure::gen;        // NOLINT
	using namespace qerasure::simulator;  // NOLINT

	constexpr uint32_t kDistance = 3;
	constexpr uint32_t kRounds = 3;
	constexpr double kErasureProb = 0.08;
	constexpr uint64_t kShots = 1;
	constexpr uint64_t kSeed = 20260304;

	SurfaceCodeRotated generator(kDistance);
	const ErasureCircuit erasure_circuit =
		generator.build_circuit(kRounds, kErasureProb, /*erasable_qubits=*/"ALL");

	ErasureModel model(
		/*max_persistence=*/2,
		/*onset=*/PauliChannel(0.25, 0.25, 0.25),
		/*reset=*/PauliChannel(0.25, 0.25, 0.25),
		/*spread=*/TQGSpreadModel(PauliChannel(0.5, 0.0, 0.0), PauliChannel(0.0, 0.0, 0.5)));
	model.check_false_negative_prob = 0.02;
	model.check_false_positive_prob = 0.0;

	const CompiledErasureProgram compiled(erasure_circuit, model);
	ErasureSampler sampler(compiled);
	SurfHMMDecoder decoder(compiled);
	Injector injector;

	SamplerParams params{};
	params.shots = kShots;
	params.seed = kSeed;
	const SampledBatch batch = sampler.sample(params);
	if (batch.shots.size() != kShots) {
		throw std::runtime_error("Unexpected sampled shot count in surf_hmm_decoder_trace_test");
	}

	for (uint32_t shot = 0; shot < batch.shots.size(); ++shot) {
		std::cout << "\n=== SHOT " << shot << " SAMPLED TRACE ===\n";
		std::cout << batch.shots[shot].to_string() << "\n";

		const std::vector<uint8_t> check_bits = extract_check_bits(batch.shots[shot]);
		if (check_bits.size() != compiled.num_checks()) {
			throw std::runtime_error("Extracted check bit count does not match compiled num_checks");
		}

		const stim::Circuit injected = injector.inject(batch, shot);
		std::cout << "=== SHOT " << shot << " LOOKBACK POSTERIORS ===\n";
		const SpreadInjectionBuckets buckets =
			decoder.compute_spread_injections(&check_bits, /*print_posteriors=*/true);
		std::cout << "=== SHOT " << shot << " SPREAD INJECTIONS ===\n";
		bool printed_event = false;
		for (uint32_t op_index = 0; op_index < buckets.size(); ++op_index) {
			for (const auto& event : buckets[op_index]) {
				printed_event = true;
				const char* op = "I";
				switch (event.operation) {
					case PauliOperation::X:
						op = "X_ERROR";
						break;
					case PauliOperation::Y:
						op = "Y_ERROR";
						break;
					case PauliOperation::Z:
						op = "Z_ERROR";
						break;
					case PauliOperation::I:
						op = "I";
						break;
				}
				std::cout << "op_index=" << op_index
						  << " target=" << event.target_qubit
						  << " op=" << op
						  << " p=" << event.probability << "\n";
			}
		}
		if (!printed_event) {
			std::cout << "(none)\n";
		}
		const stim::Circuit decoded = decoder.decode(injected, &check_bits, /*print_posteriors=*/false);
		std::cout << "=== SHOT " << shot << " DECODED CIRCUIT ===\n";
		std::cout << decoded.str() << "\n";
	}

	return 0;
}
