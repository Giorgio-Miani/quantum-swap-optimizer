import sys
import os
import time
import pandas as pd

# Manually specify the path to the 'src' directory
src_path = os.path.abspath(os.path.join(os.getcwd(), './src'))

# Add the specified path to the system path
sys.path.append(src_path)

# Local application/library imports
import backend_gen as backendGen
import circuit_gen as circuitGen
import qubit_mapping as qMap

num_modules        = 6
module_max_qubits  = 4
module_max_gates   = 10
reduced_distance   = None
max_allowed_weight = 5
num_qubits_x       = 100
num_qubits_y       = 100
heuristic          = False
save_backend       = False

results = pd.DataFrame(columns=['seed',
                                'time',
                                'circuit_depth',
                                'q_map_depth', 
                                'circuit_total_qubits', 
                                'q_map_total_qubits',
                                'circuit_gate_count', 
                                'q_map_gate_count',
                                'circuit_swap_count',
                                'q_map_swap_count',
                                'circuit_t_count', 
                                'q_map_t_count',
                                'circuit_t_depth', 
                                'q_map_t_depth'])

backend = backendGen.generate_regular_backend(num_qubits_x, num_qubits_y)

print(f"=====================================================")

for seed in range(100, 111):
    start_time = time.time()

    print(f'Generating circuit with seed {seed}')
    circuit = circuitGen.RandomCircuit(
        num_modules, 
        module_max_qubits, 
        module_max_gates, 
        seed
    )
    circuit.gen_random_circuit()

    print('Finding Qubit Mapping')
    q_map = qMap.QubitMapping(
        circuit, 
        backend=backend,
        coupling_map_dims=(num_qubits_x, num_qubits_y),
        reduced_distance=reduced_distance, 
        max_allowed_weight=max_allowed_weight,
        heuristic=heuristic
    )
    q_map.generate_ASAP_qubit_mapping()

    end_time = time.time()
    total_time = round(end_time - start_time, 1)

    print(f"Qubit modules: {q_map.modules_qubits}")
    print(f"Qubit mapping: {q_map.qubit_mapping}")
    print(f"Computing metrics")

    qiskit_metrics = circuit.get_benchmark_metrics(
        backend=q_map.backend,
        coupling_map=q_map.backend.coupling_map
    )
    q_map_metrics  = q_map.benchmark_metrics

    result = [seed, total_time]
    for field, bench_value in qiskit_metrics.items():
        q_map_value = q_map_metrics[field]
        result.append(bench_value)
        result.append(q_map_value)
    results.loc[len(results)] = result

    print(f"=====================================================")

os.makedirs('./output', exist_ok=True)
results.to_csv(f'./output/nm{num_modules}_mmq{module_max_qubits}_mmg{module_max_gates}_rd{reduced_distance}_maw{max_allowed_weight}_nqx{num_qubits_x}_nqy{num_qubits_y}_h{heuristic}.csv')