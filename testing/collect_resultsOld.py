import sys
import os
import pickle
import random
import time
import pandas as pd

# Manually specify the path to the 'src' directory
src_path = os.path.abspath(os.path.join(os.getcwd(), './src'))

# Add the specified path to the system path
sys.path.append(src_path)

# Third-party libraries
import networkx as nx
from qiskit import transpile
import mapomatic as mm

# Local application/library imports
import src.backend_gen as backendGen
import src.circuit_gen as circuitGen
import src.qubit_mapping as qMap

num_modules        = 4
module_max_qubits  = 4
module_max_gates   = 6
buffer_distance    = 0
reduced_distance   = None
max_allowed_weight = 3
num_qubits_x       = 9
num_qubits_y       = 9
heuristic          = False
save_backend       = False

results = pd.DataFrame(columns=['seed','num_modules','module_max_qubits','module_max_gates','buffer_distance','reduced_distance','max_allowed_weight','num_qubits_x','num_qubits_y','circuit_depth', 'q_map_depth', 'circuit_total_qutbis', 'q_map_total_qubits','circuit_gate_count', 'q_map_gate_count','circuit_t_count', 'q_map_t_count','circuit_t_depth', 'q_map_t_depth', 'time'])

backend = backendGen.generate_regular_backend(num_qubits_x, num_qubits_y)

for seed in range(1,11):

    start_time = time.time()

    print(f'Circuit with seed {seed}')

    print('Generating circuit...')
    # Generate random circuit
    circuit = circuitGen.RandomCircuit(num_modules, module_max_qubits, module_max_gates, seed)
    circuit.gen_random_circuit()

    # Ploy dependency graph
    circuit.draw_dependency_graph()

    print('Finding Qubit Mapping')
    # Generate the Quantum Circuit Mapping
    q_map = qMap.QubitMapping(
        circuit, 
        backend=backend,
        buffer_distance=buffer_distance, 
        reduced_distance=reduced_distance, 
        max_allowed_weight=max_allowed_weight,
        heuristic=heuristic
    )
    q_map.generate_ALAP_qubit_mapping()

    end_time = time.time()
    total_time = round(end_time - start_time, 1)

    print(f"Qubit modules:  {q_map.modules_qubits}")
    print(f"Qubit mapping: {q_map.qubit_mapping}")

    print('Calculating metrics')

    benchmarck_metrics = circuit.get_benchmark_metrics(coupling_map=backend.coupling_map)
    q_map_metrics      = q_map.benchmark_metrics

    result = [seed, num_modules, module_max_qubits, module_max_gates, buffer_distance, reduced_distance, max_allowed_weight, num_qubits_x, num_qubits_y]

    for field, bench_value in benchmarck_metrics.items():
        q_map_value = q_map_metrics[field]
        result.append(bench_value)
        result.append(q_map_value)

    result.append(total_time)

    results.loc[len(results)] = result


results.to_csv(f'./results/nm{num_modules}_mmq{module_max_qubits}_mmg{module_max_gates}_bd{buffer_distance}_rd{reduced_distance}_maw{max_allowed_weight}_nqx{num_qubits_x}_nqy{num_qubits_y}.csv')
