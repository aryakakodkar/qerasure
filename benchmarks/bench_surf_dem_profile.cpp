#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include "core/circuit/compile.h"
#include "core/circuit/erasure_model.h"
#include "core/decode/surf_dem_builder.h"
#include "core/gen/surf.h"
#include "core/model/pauli_channel.h"
#include "core/simulator/sim_internal_utils.h"
#include "core/simulator/stream_sampler.h"
#include "stim/util_top/circuit_to_dem.h"

namespace {

using qerasure::decode::SkippableReweightMap;
using qerasure::decode::SpreadInjectionBuckets;

uint64_t make_op_qubit_key(uint32_t op_index, uint32_t qubit) {
	return (static_cast<uint64_t>(op_index) << 32) | qubit;
}

stim::Circuit build_decoded_circuit_from_buckets(
	const qerasure::circuit::CompiledErasureProgram& program,
	const SpreadInjectionBuckets& buckets,
	const SkippableReweightMap& skippable_reweights) {
	stim::Circuit injected;
	for (uint32_t op_index = 0; op_index < program.operation_groups.size(); ++op_index) {
		const qerasure::circuit::OperationGroup& op_group = program.operation_groups[op_index];
		if (op_group.stim_instruction.has_value()) {
			const qerasure::circuit::Instruction& instr = op_group.stim_instruction.value();
			const bool should_reweight = qerasure::circuit::is_erasure_skippable_op(instr.op) &&
										 qerasure::circuit::is_probabilistic_op(instr.op);
			if (!should_reweight) {
				qerasure::simulator::internal::append_mapped_stim_instruction(instr, &injected);
			} else {
				const char* op_name = qerasure::circuit::opcode_name(instr.op);
				for (const uint32_t target : instr.targets) {
					double p_unerased = 1.0;
					const auto it = skippable_reweights.find(make_op_qubit_key(op_index, target));
					if (it != skippable_reweights.end()) {
						p_unerased = it->second;
					}
					const double reweighted_prob = std::clamp(instr.arg * p_unerased, 0.0, 1.0);
					if (reweighted_prob <= 0.0) {
						continue;
					}
					injected.safe_append_ua(op_name, {target}, reweighted_prob);
				}
			}
		}

		for (const auto& event : buckets[op_index]) {
			const double p_x = std::clamp(event.p_x, 0.0, 1.0);
			const double p_y = std::clamp(event.p_y, 0.0, 1.0);
			const double p_z = std::clamp(event.p_z, 0.0, 1.0);
			if (p_x > 0.0 || p_y > 0.0 || p_z > 0.0) {
				injected.safe_append_u("PAULI_CHANNEL_1", {event.target_qubit}, {p_x, p_y, p_z});
			}
		}
	}
	return injected;
}

}  // namespace

int main(int argc, char** argv) {
	using namespace qerasure::circuit;    // NOLINT
	using namespace qerasure::decode;     // NOLINT
	using namespace qerasure::gen;        // NOLINT
	using namespace qerasure::simulator;  // NOLINT

	uint32_t shots = 5000;
	uint32_t sampler_threads = std::thread::hardware_concurrency();
	if (sampler_threads == 0) {
		sampler_threads = 1;
	}
	if (argc > 1) {
		shots = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
	}
	if (argc > 2) {
		sampler_threads = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
	}

	constexpr uint32_t kDistance = 15;
	constexpr uint32_t kRounds = 15;
	constexpr double kErasureProb = 0.01;
	constexpr uint32_t kSeed = 12345;

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
	SurfDemBuilder decoder(compiled);

	std::vector<std::vector<uint8_t>> check_rows(shots);
	std::atomic<uint32_t> row_index{0};
	sampler.sample_with_callback(
		shots,
		kSeed,
		[&](const stim::Circuit&, const std::vector<uint8_t>& check_results) {
			const uint32_t i = row_index.fetch_add(1, std::memory_order_relaxed);
			if (i < shots) {
				check_rows[i] = check_results;
			}
		},
		sampler_threads);

	stim::DemOptions dem_options{};
	dem_options.decompose_errors = false;
	dem_options.flatten_loops = true;
	dem_options.allow_gauge_detectors = true;
	dem_options.approximate_disjoint_errors_threshold = 1.0;
	dem_options.ignore_decomposition_failures = true;

	uint64_t compute_ns = 0;
	uint64_t build_ns = 0;
	uint64_t dem_ns = 0;
	uint64_t bucket_sink = 0;
	uint64_t op_sink = 0;
	uint64_t dem_error_sink = 0;

	for (const auto& check_results : check_rows) {
		SkippableReweightMap skippable_reweights;
		const auto t0 = std::chrono::steady_clock::now();
		SpreadInjectionBuckets buckets =
			decoder.compute_spread_injections(&check_results, /*verbose=*/false, &skippable_reweights);
		const auto t1 = std::chrono::steady_clock::now();
		stim::Circuit decoded = build_decoded_circuit_from_buckets(compiled, buckets, skippable_reweights);
		const auto t2 = std::chrono::steady_clock::now();
		stim::DetectorErrorModel dem = stim::circuit_to_dem(decoded, dem_options);
		const auto t3 = std::chrono::steady_clock::now();

		compute_ns += static_cast<uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
		build_ns += static_cast<uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
		dem_ns += static_cast<uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());

		for (const auto& bucket : buckets) {
			bucket_sink += bucket.size();
		}
		op_sink += decoded.operations.size();
		dem_error_sink += dem.count_errors();
	}

	const double compute_s = static_cast<double>(compute_ns) / 1e9;
	const double build_s = static_cast<double>(build_ns) / 1e9;
	const double dem_s = static_cast<double>(dem_ns) / 1e9;
	const double total_s = compute_s + build_s + dem_s;
	const double us_total_s = compute_s + build_s;

	auto pct = [total_s](double value_s) {
		return total_s > 0.0 ? (100.0 * value_s / total_s) : 0.0;
	};

	std::cout << "bench_surf_dem_profile\n";
	std::cout << "distance: " << kDistance << "\n";
	std::cout << "rounds: " << kRounds << "\n";
	std::cout << "erasure_prob: " << kErasureProb << "\n";
	std::cout << "shots: " << shots << "\n";
	std::cout << "sampler_threads: " << sampler_threads << "\n";
	std::cout << "compute_spread_s: " << compute_s << " (" << pct(compute_s) << "%)\n";
	std::cout << "circuit_build_s: " << build_s << " (" << pct(build_s) << "%)\n";
	std::cout << "dem_build_s: " << dem_s << " (" << pct(dem_s) << "%)\n";
	std::cout << "our_total_s: " << us_total_s << " (" << pct(us_total_s) << "%)\n";
	std::cout << "profile_total_s: " << total_s << "\n";
	std::cout << "per_shot_compute_ms: " << (shots > 0 ? 1e3 * compute_s / shots : 0.0) << "\n";
	std::cout << "per_shot_circuit_build_ms: " << (shots > 0 ? 1e3 * build_s / shots : 0.0) << "\n";
	std::cout << "per_shot_dem_build_ms: " << (shots > 0 ? 1e3 * dem_s / shots : 0.0) << "\n";
	std::cout << "bucket_sink: " << bucket_sink << "\n";
	std::cout << "op_sink: " << op_sink << "\n";
	std::cout << "dem_error_sink: " << dem_error_sink << "\n";

	return EXIT_SUCCESS;
}
