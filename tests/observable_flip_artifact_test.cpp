#include <algorithm>
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
#include "core/simulator/stream_sampler.h"

int main() {
	using namespace qerasure::circuit;    // NOLINT
	using namespace qerasure::decode;     // NOLINT
	using namespace qerasure::gen;        // NOLINT
	using namespace qerasure::simulator;  // NOLINT

	constexpr uint32_t kDistance = 3;
	constexpr uint32_t kRounds = 3;
	constexpr double kErasureProb = 0.08;
	constexpr uint32_t kMaxTries = 10'000;
	constexpr uint32_t kSeedBase = 20260306;

	SurfaceCodeRotated generator(kDistance);
	const ErasureCircuit erasure_circuit =
		generator.build_circuit(kRounds, kErasureProb, /*erasable_qubits=*/"ALL");

	ErasureModel model(
		/*max_persistence=*/2,
		/*onset=*/PauliChannel(0.25, 0.25, 0.25),
		/*reset=*/PauliChannel(0.25, 0.25, 0.25),
		/*spread=*/TQGSpreadModel(PauliChannel(0.5, 0.0, 0.0), PauliChannel(0.0, 0.0, 0.5)));
	model.check_false_negative_prob = 0.0;
	model.check_false_positive_prob = 0.0;

	const CompiledErasureProgram compiled(erasure_circuit, model);
	StreamSampler sampler(compiled);
	SurfDemBuilder decoder(compiled);

	const std::filesystem::path output_dir =
		std::filesystem::path("tests") / "artifacts" / "observable_flip_example";
	std::filesystem::create_directories(output_dir);

	bool found = false;
	uint32_t found_seed = 0;
	stim::Circuit found_logical_circuit;
	stim::Circuit found_decoded_circuit;
	std::vector<uint8_t> found_check_bits;
	std::ostringstream posterior_stream;
	std::vector<uint8_t> found_observable_row;

	for (uint32_t attempt = 0; attempt < kMaxTries; ++attempt) {
		const uint32_t seed = kSeedBase + attempt;
		const SyndromeSampleBatch sampled = sampler.sample_syndromes(/*num_shots=*/1, seed, /*threads=*/1);
		if (sampled.num_observables == 0) {
			throw std::runtime_error("No observables found in sampled logical circuit.");
		}

		bool has_flip = false;
		for (uint32_t o = 0; o < sampled.num_observables; ++o) {
			if (sampled.observable_flips[o] != 0) {
				has_flip = true;
				break;
			}
		}
		if (!has_flip) {
			continue;
		}

		stim::Circuit logical_circuit;
		std::vector<uint8_t> check_bits;
		sampler.sample_with_callback(
			/*num_shots=*/1,
			seed,
			[&](const stim::Circuit& logical, const std::vector<uint8_t>& checks) {
				logical_circuit = logical;
				check_bits = checks;
			},
			/*num_threads=*/1);

		if (check_bits.size() != sampled.num_checks) {
			throw std::runtime_error("Check bit width mismatch while extracting observable-flip artifact.");
		}
		for (uint32_t i = 0; i < sampled.num_checks; ++i) {
			if (check_bits[i] != sampled.check_flags[i]) {
				throw std::runtime_error(
					"check_flags mismatch between sample_syndromes and sample_with_callback.");
			}
		}

		std::streambuf* old_cout = std::cout.rdbuf(posterior_stream.rdbuf());
		const stim::Circuit decoded_circuit =
			decoder.build_decoded_circuit(&check_bits, /*verbose=*/true);
		std::cout.rdbuf(old_cout);

		found = true;
		found_seed = seed;
		found_logical_circuit = logical_circuit;
		found_decoded_circuit = decoded_circuit;
		found_check_bits = check_bits;
		found_observable_row.assign(
			sampled.observable_flips.begin(),
			sampled.observable_flips.begin() + sampled.num_observables);
		break;
	}

	if (!found) {
		throw std::runtime_error(
			"Failed to find an observable-flip shot within max tries. "
			"Increase kMaxTries or tune noise parameters.");
	}

	const std::filesystem::path logical_path = output_dir / "logical_circuit.stim";
	const std::filesystem::path decoded_path = output_dir / "decoded_circuit.stim";
	const std::filesystem::path erasure_path = output_dir / "erasure_circuit.qer";
	const std::filesystem::path posteriors_path = output_dir / "posteriors_verbose.txt";
	const std::filesystem::path metadata_path = output_dir / "metadata.txt";

	{
		std::ofstream out(erasure_path);
		if (!out) {
			throw std::runtime_error("Failed to open erasure_circuit artifact for writing.");
		}
		out << erasure_circuit.to_string();
	}
	{
		std::ofstream out(logical_path);
		if (!out) {
			throw std::runtime_error("Failed to open logical_circuit artifact for writing.");
		}
		out << found_logical_circuit.str();
	}
	{
		std::ofstream out(decoded_path);
		if (!out) {
			throw std::runtime_error("Failed to open decoded_circuit artifact for writing.");
		}
		out << found_decoded_circuit.str();
	}
	{
		std::ofstream out(posteriors_path);
		if (!out) {
			throw std::runtime_error("Failed to open posteriors_verbose artifact for writing.");
		}
		out << posterior_stream.str();
	}
	{
		std::ofstream out(metadata_path);
		if (!out) {
			throw std::runtime_error("Failed to open metadata artifact for writing.");
		}
		out << "seed=" << found_seed << "\n";
		out << "observable_row=";
		for (size_t i = 0; i < found_observable_row.size(); ++i) {
			if (i > 0) {
				out << ",";
			}
			out << static_cast<uint32_t>(found_observable_row[i]);
		}
		out << "\n";
		out << "check_bits_count=" << found_check_bits.size() << "\n";
	}

	std::cout << "observable_flip_artifact_test\n";
	std::cout << "seed: " << found_seed << "\n";
	std::cout << "saved: " << erasure_path << "\n";
	std::cout << "saved: " << logical_path << "\n";
	std::cout << "saved: " << decoded_path << "\n";
	std::cout << "saved: " << posteriors_path << "\n";
	std::cout << "saved: " << metadata_path << "\n";

	return 0;
}
