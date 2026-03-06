#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/circuit/compile.h"
#include "core/circuit/erasure_model.h"
#include "core/decode/surf_dem_builder.h"
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
	SurfDemBuilder decoder(compiled);
	Injector injector;
	const std::filesystem::path output_dir = "surf_dem_builder_trace_outputs";
	std::filesystem::create_directories(output_dir);

	SamplerParams params{};
	params.shots = kShots;
	params.seed = kSeed;
	const SampledBatch batch = sampler.sample(params);
	if (batch.shots.size() != kShots) {
		throw std::runtime_error("Unexpected sampled shot count in surf_dem_builder_trace_test");
	}

	for (uint32_t shot = 0; shot < batch.shots.size(); ++shot) {
		const std::vector<uint8_t> check_bits = extract_check_bits(batch.shots[shot]);
		if (check_bits.size() != compiled.num_checks()) {
			throw std::runtime_error("Extracted check bit count does not match compiled num_checks");
		}

		const stim::Circuit sampled_injected = injector.inject(batch, shot);

		// Capture posterior debug output produced by decoder when verbose=true.
		std::ostringstream posterior_stream;
		std::streambuf* old_cout = std::cout.rdbuf(posterior_stream.rdbuf());
		const stim::Circuit decoded_injected =
			decoder.build_decoded_circuit(&check_bits, /*verbose=*/true);
		std::cout.rdbuf(old_cout);

		const std::filesystem::path sampled_trace_path =
			output_dir / ("shot_" + std::to_string(shot) + "_sampled_trace.txt");
		const std::filesystem::path posterior_path =
			output_dir / ("shot_" + std::to_string(shot) + "_posteriors.txt");
		const std::filesystem::path circuit_path =
			output_dir / ("shot_" + std::to_string(shot) + "_circuit.stim");

		{
			std::ofstream out(sampled_trace_path);
			if (!out) {
				throw std::runtime_error("Failed to open sampled trace output file");
			}
			out << batch.shots[shot].to_string();
		}

		{
			std::ofstream out(posterior_path);
			if (!out) {
				throw std::runtime_error("Failed to open posterior output file");
			}
			out << posterior_stream.str();
		}

		{
			std::ofstream out(circuit_path);
			if (!out) {
				throw std::runtime_error("Failed to open circuit output file");
			}
			out << decoded_injected.str();
		}

		std::cout << "Saved shot " << shot << " outputs:\n";
		std::cout << "  " << sampled_trace_path << "\n";
		std::cout << "  " << posterior_path << "\n";
		std::cout << "  " << circuit_path << "\n";
	}

	return 0;
}
