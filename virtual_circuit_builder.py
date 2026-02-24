from pathlib import Path
import sys

# Required to import qerasure from the local repository
repo_root = Path.cwd().resolve().parent / 'qerasure'
python_src = repo_root / 'python'

if not python_src.exists():
    raise FileNotFoundError(f'Expected qerasure python source at {python_src}')

if str(python_src) not in sys.path:
    sys.path.insert(0, str(python_src))

print(f'Added to sys.path: {python_src}')

import qerasure
import stim
from dataclasses import dataclass

############ ERASURE CIRCUIT SIMULATION AND LOWERING ############

DISTANCE = 3
QEC_ROUNDS = 3
SHOTS = 10

P_TQE = 0.01 # probability of two-qubit erasure error

# Create a rotated surface code object
code = qerasure.RotatedSurfaceCode(DISTANCE)

# Build the noise parameters for the simulation
noise = qerasure.NoiseParams()
noise.set(qerasure.NoiseChannel.TWO_QUBIT_ERASURE, P_TQE)

# Set the parameters for the erasure simulation

sim_params = qerasure.ErasureSimParams(
    code=code,
    noise=noise,
    qec_rounds=QEC_ROUNDS,
    shots=SHOTS,
    seed=12345,
    erasure_selection=qerasure.ErasureQubitSelection.DATA_QUBITS # only allow erasure on data qubits
)

erasure_simulator = qerasure.ErasureSimulator(sim_params)
erasure_results = erasure_simulator.simulate()

# Configure lowering parameters
program = qerasure.SpreadProgram()
# 0.5 probability Z-error on the two X-ancillas
program.add_error_channel(0.5, [qerasure.SpreadTargetOp(qerasure.PauliError.Z_ERROR, qerasure.PartnerSlot.X_1)])
program.add_error_channel(0.5, [qerasure.SpreadTargetOp(qerasure.PauliError.Z_ERROR, qerasure.PartnerSlot.X_2)])
# 0.5 probability X-error on first Z-ancilla
program.add_correlated_error(0.5, [qerasure.SpreadTargetOp(qerasure.PauliError.X_ERROR, qerasure.PartnerSlot.Z_1)]) 
# certain X-error on second Z-ancilla, but only if the previous operation was not sampled
program.add_else_correlated_error(1.0, [qerasure.SpreadTargetOp(qerasure.PauliError.X_ERROR, qerasure.PartnerSlot.Z_2)])

reset = qerasure.LoweredErrorParams(qerasure.PauliError.DEPOLARIZE, 0.75) # probability of depolarizing error on reset
lowering_params = qerasure.LoweringParams(program, reset)

lowerer = qerasure.Lowerer(code, lowering_params)
lowering_result = lowerer.lower(erasure_results)

@dataclass
class VirtualError:
    event_type: qerasure.PauliError
    qubit_idx: int
    timestep: int

################## YALE VIRTUAL CIRCUIT ###############
# This virtual circuit is based on the protocol from the Yale paper

for shot in range(SHOTS):
    print(f"Processing shot {shot}")
    logical_circuit = qerasure.build_logical_stabilizer_circuit_object(code=code, lowering_result=lowering_result, shot_index=shot)

    cur_offset = 0
    for event_number, event in enumerate(erasure_results.sparse_erasures[shot]):
        # Figure out which timestep each event corresponds to
        for i in erasure_results.erasure_timestep_offsets[shot][cur_offset:]:
            if event_number + 1 <= i:
                break
                
            cur_offset += 1

        if event.event_type == qerasure.EventType.RESET:
            round = cur_offset // 4

            instruction_probs = [[0, 0, 0, 0] for _ in range(len(program.instructions))]
            for idx, instruction in enumerate(program.instructions):
                
                instruction_min = -1
                correlated_probability = 1

                # find the minimum timestep for any instruction to be possible (i.e. the minimum timestep at which this data qubit is gated with an ancilla in that instruction)
                for target in instruction.targets:
                    # check that the instruction is relevant for this qubit (i.e. it involves an ancilla that is gated with this qubit)
                    try:
                        timestep_for_data_slot = code.timestep_for_data_slot(event.qubit_idx, target.slot)
                    except ValueError:
                        continue

                    if instruction_min == -1 or timestep_for_data_slot < instruction_min:
                        instruction_min = timestep_for_data_slot

                # figure out what the probability is that this instruction is applied if the reset error occurs at a given timestep
                for timestep in range(4):
                    if instruction_min < timestep:
                        continue
                    
                    p = 0
                    if instruction.type == qerasure.SpreadInstructionType.ERROR_CHANNEL:
                        p = instruction.probability
                    elif instruction.type == qerasure.SpreadInstructionType.CORRELATED_ERROR:
                        p = instruction.probability
                        correlated_probability = 1 - p
                    elif instruction.type == qerasure.SpreadInstructionType.ELSE_CORRELATED_ERROR:
                        p = instruction.probability * correlated_probability
                        correlated_probability *= (1 - instruction.probability)

                    instruction_probs[idx][timestep] = p

            # ith element of this array is the probability that an erasure first occurs after the ith gate step
            occurrence_probs = [(1-P_TQE)**(i - 1) * P_TQE for i in range(1, 5)]

            if shot == 0:
                p_instructions = [sum(instruction_probs[idx][timestep] * occurrence_probs[timestep] for timestep in range(4)) for idx in range(len(program.instructions))]
                print(f"Qubit {event.qubit_idx} is reset at time {cur_offset - 1} (round {round})")

                for idx, instruction in enumerate(program.instructions):
                    print(f"Instruction {idx}: {instruction.type} with targets {instruction.targets} has probabilities {p_instructions[idx]} for this event")
    
    # det_sampler = logical_circuit.compile_detector_sampler()
    # print(det_sampler.sample(shots=1))
