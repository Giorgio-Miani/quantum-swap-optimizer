# Generic libraries imports
import random
import time
import os
import pandas as pd
from datetime import datetime
import tracemalloc
import multiprocessing
import sys

src_path = os.path.abspath(os.path.join(os.getcwd(), './src'))
# Add the specified path to the system path
sys.path.append(src_path)

# Local application/library imports
import src.backend_gen as backendGen
import src.circuit_gen as circuitGen
import src.qubit_mapping as qMap


def generate_results():
    tracemalloc.start()
    iteration = 10
    timeout_duration = 60  # Maximum allowed time per result generation in seconds

    for i in range(iteration):
        start_time = time.time()

        try:
            # Create a separate process to execute generateOneResult
            process = multiprocessing.Process(target=generate_one_result, args=(i,))
            process.start()

            # Monitor the process and terminate it if it exceeds the timeout
            process.join(timeout_duration)
            if process.is_alive():
                process.terminate()  # Terminate the process if it exceeds the timeout
                process.join()  # Ensure the process is fully terminated
                print(f"\033[93mIteration {i} exceeded {timeout_duration} seconds and was cancelled.\033[0m")
            else:
                end_time = time.time()
                current, peak = tracemalloc.get_traced_memory()
                print(
                    f"\033[92mIteration {i} completed in {end_time - start_time:.2f} seconds. Memory usage: {current / 10 ** 6:.2f} MB; Peak: {peak / 10 ** 6:.2f} MB\033[0m")

        except Exception as e:
            # Handle any exceptions that occur during the execution of generateOneResult
            print(f"\033[91mError at iteration {i}: {e}\033[0m")

    tracemalloc.stop()

def generate_one_result(iteration_number):
    general_info = pd.DataFrame(columns=[
        'name',
        'date_of_execution',
        'duration_of_execution',
        'seed',
        'number_of_modules',
        'number_of_max_qubits_per_module',
        'number_of_max_gates_per_module',
        'reduced_distance',
        'maximum_allowed_weight',
        'number_of_qubits_x_[row]',
        'number_of_qubits_y_[column]',
        'heuristic',
        'qubit_modules',
        'qubit_mapping'
    ])
    general_info = pd.concat([general_info, pd.DataFrame([pd.Series(dtype='object')])], ignore_index=True)  # Just creating a row and filling with NaN
    general_info.at[0, 'date_of_execution'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    our_performance = pd.DataFrame(columns=[
        'depth',
        'total_qubits',
        'gate_count',
        'swap_count',
        't_count'
    ])
    qiskit_performance = pd.DataFrame(columns=[
        'qiskit_optimization_level',
        'qiskit_depth',
        'qiskit_total_qubits',
        'qiskit_gate_count',
        'qiskit_swap_count',
        'qiskit_t_count'
    ])

    workingDir = 'result' + str(iteration_number)

    general_info.at[0, 'name'] = workingDir

    # Parameters
    num_modules = 6
    module_max_qubits = 4
    module_max_gates = 6
    reduced_distance = None
    max_allowed_weight = 5
    num_qubits_x = 100
    num_qubits_y = 100
    heuristic = False
    save_backend = False
    seed = random.randint(1, int(1e4))

    dir = f'results/nm{num_modules}_mmq{module_max_qubits}_mmg{module_max_gates}_rd{reduced_distance}_maw{max_allowed_weight}_nqx{num_qubits_x}_nqy{num_qubits_y}_h{heuristic}/'
    workingDir = dir + workingDir
    os.makedirs(workingDir, exist_ok=True)

    print('Generating ' + workingDir + ':')

    # %% 1 # failing_seed result file creation and seed saving
    general_info.at[0, 'seed'] = seed

    failing_seed_path = os.path.join(workingDir, 'failing_seed.txt')
    with open(failing_seed_path, 'w') as file:
        file.write(str(seed) + '\n')

    # %% 2 # Random circuit generation:
    start_time = time.time()
    circuit = circuitGen.RandomCircuit(num_modules, module_max_qubits, module_max_gates,
                                       seed)  # Defining circuit properties
    circuit.gen_random_circuit()  # Actually generating the circuit with the properties specified above
    end_time = time.time()
    total_time = end_time - start_time

    # %% 3 # Generating the quantum circuit mapping
    start_time = time.time()
    ourBackend = backendGen.generate_regular_backend(num_qubits_x, num_qubits_y)
    q_map = qMap.QubitMapping(  # Defining mapping properties and constrains
        circuit,
        backend=ourBackend,
        coupling_map_dims=(num_qubits_x, num_qubits_y),
        reduced_distance=reduced_distance,
        max_allowed_weight=max_allowed_weight,
        heuristic=heuristic
    )
    q_map.generate_ASAP_qubit_mapping()  # Actually generating the mapping with the properties specified above using an ASAP approach
    end_time = time.time()
    total_time += (end_time - start_time)

    # %% 4 # Retrieving Qiskit metrics
    # For all 4 possible optimizations (0, 1, 2 or 3)
    for i in range(4):
        qiskit_metrics = circuit.get_benchmark_metrics(
            backend=q_map.backend,
            coupling_map=q_map.backend.coupling_map,
            optimization_level=i
        )
        qiskit_metrics_prefixed = {f"qiskit_{k}": v for k, v in
                                   qiskit_metrics.items()}  # just adding "qiskit" prefix on the key
        qiskit_performance = pd.concat([qiskit_performance, pd.DataFrame([qiskit_metrics_prefixed])],
                                       ignore_index=True)  # saving the metrics into our dataset

    # %% 5 # Retrieving our mapping metrics
    our_metrics = q_map.benchmark_metrics
    our_performance = pd.concat([our_performance, pd.DataFrame([our_metrics])],
                                ignore_index=True)  # saving the metrics into our dataset

    # %% 6 # Saving the quantum circuit parameters
    general_info.at[0, 'duration_of_execution'] = total_time
    general_info.at[0, 'number_of_modules'] = num_modules
    general_info.at[0, 'number_of_max_qubits_per_module'] = module_max_qubits
    general_info.at[0, 'number_of_max_gates_per_module'] = module_max_gates
    general_info.at[0, 'reduced_distance'] = reduced_distance
    general_info.at[0, 'maximum_allowed_weight'] = max_allowed_weight
    general_info.at[0, 'number_of_qubits_x_[row]'] = num_qubits_x
    general_info.at[0, 'number_of_qubits_y_[column]'] = num_qubits_y
    general_info.at[0, 'heuristic'] = heuristic
    general_info.at[0, 'qubit_modules'] = q_map.modules_qubits
    general_info.at[0, 'qubit_mapping'] = q_map.qubit_mapping

    # %% 7 Finally saving everything on disk:

    # unifying dataframes on the same row
    combined_data = pd.concat([general_info, our_performance, qiskit_performance], axis=1)
        # csv format (useful for importing into spreadsheets softwares)
    if os.path.exists(dir + 'total_metrics.csv'):
        collected_results = pd.read_csv(dir + 'total_metrics.csv', index_col=None)
        collected_results = pd.concat([collected_results, combined_data], ignore_index=True)
    else:
        collected_results = combined_data
    combined_data.to_csv(os.path.join(workingDir, 'metrics.csv'), index=False)
    collected_results.to_csv(dir + 'total_metrics.csv', index=False)
        # txt format (human readable)
    with open(os.path.join(workingDir, 'metrics.txt'), 'w') as file:
        file.write("### Results: ###\n")
        file.write(combined_data.to_string(index=False))

    # If everything worked correctly we can delete the failing_seed file. If the process is killed or throws an execption
    #   the file remains and we can see the seed later.
    if os.path.exists(failing_seed_path):
        os.remove(failing_seed_path)
        print(f"\033[92m{workingDir} generated successfully\033[0m")
    else:
        print(f"\033[91mUnexpected error in failing_seed removal\033[0m")



if __name__ == '__main__':
    generate_results()
