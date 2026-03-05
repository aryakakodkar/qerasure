#include <atomic>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

#include "core/circuit/compile.h"
#include "core/circuit/erasure_model.h"
#include "core/decode/surf_hmm_decoder.h"
#include "core/gen/surf.h"
#include "core/model/pauli_channel.h"
#include "core/simulator/stream_sampler.h"
#include "stim/util_top/circuit_to_dem.h"

int main() {
	using namespace qerasure::circuit;    // NOLINT
	using namespace qerasure::decode;     // NOLINT
	using namespace qerasure::gen;        // NOLINT
	using namespace qerasure::simulator;  // NOLINT

	constexpr uint32_t kDistance = 15;
	constexpr uint32_t kRounds = 15;
	constexpr double kErasureProb = 0.01;
	constexpr uint32_t kShots = 100;
	constexpr uint32_t kSeed = 12345;
	constexpr uint32_t kThreads = 4;

	SurfaceCodeRotated generator(kDistance);
	const ErasureCircuit erasure_circuit =
		generator.build_circuit(kRounds, kErasureProb, /*erasable_qubits=*/"ALL");

	ErasureModel model(
		/*max_persistence=*/2,
		/*onset=*/PauliChannel(0.18, 0.06, 0.06),
		/*reset=*/PauliChannel(0.04, 0.03, 0.03),
		/*spread=*/TQGSpreadModel(PauliChannel(0.12, 0.04, 0.04), PauliChannel(0.10, 0.05, 0.05)));
	model.check_false_negative_prob = 0.02;
	model.check_false_positive_prob = 0.01;

	const CompiledErasureProgram compiled(erasure_circuit, model);
	StreamSampler sampler(compiled);
	SurfHMMDecoder decoder(compiled);

	std::atomic<uint32_t> shots_seen{0};
	std::atomic<uint32_t> dem_failures{0};
	std::atomic<uint64_t> total_logical_dem_errors{0};
	std::atomic<uint64_t> total_decoded_dem_errors{0};
	std::mutex first_error_mutex;
	std::string first_error_message;

	stim::DemOptions dem_options{};
	dem_options.decompose_errors = false;
	dem_options.flatten_loops = true;
	dem_options.allow_gauge_detectors = true;
	dem_options.approximate_disjoint_errors_threshold = 1.0;
	dem_options.ignore_decomposition_failures = true;

	sampler.sample(
		kShots, kSeed,
		[&](const stim::Circuit& logical_circuit, const std::vector<uint8_t>& check_results) {
			try {
				const stim::Circuit decoded_circuit =
					decoder.decode(logical_circuit, &check_results, /*print_posteriors=*/false);
				const stim::DetectorErrorModel logical_dem =
					stim::circuit_to_dem(logical_circuit, dem_options);
				const stim::DetectorErrorModel decoded_dem =
					stim::circuit_to_dem(decoded_circuit, dem_options);
				total_logical_dem_errors.fetch_add(logical_dem.count_errors(), std::memory_order_relaxed);
				total_decoded_dem_errors.fetch_add(decoded_dem.count_errors(), std::memory_order_relaxed);
			} catch (const std::exception& ex) {
				dem_failures.fetch_add(1, std::memory_order_relaxed);
				std::lock_guard<std::mutex> lock(first_error_mutex);
				if (first_error_message.empty()) {
					first_error_message = ex.what();
				}
			}
			shots_seen.fetch_add(1, std::memory_order_relaxed);
		},
		kThreads);

	const uint32_t seen = shots_seen.load(std::memory_order_relaxed);
	const uint32_t failures = dem_failures.load(std::memory_order_relaxed);
	if (seen != kShots) {
		throw std::runtime_error("stream_decode_dem_test did not process expected number of shots");
	}
	if (failures != 0) {
		throw std::runtime_error(
			"DEM construction failed for one or more shots. First error: " + first_error_message);
	}

	std::cout << "stream_decode_dem_test\n";
	std::cout << "shots_seen: " << seen << "\n";
	std::cout << "dem_failures: " << failures << "\n";
	std::cout << "logical_dem_errors_total: "
			  << total_logical_dem_errors.load(std::memory_order_relaxed) << "\n";
	std::cout << "decoded_dem_errors_total: "
			  << total_decoded_dem_errors.load(std::memory_order_relaxed) << "\n";
	return 0;
}
