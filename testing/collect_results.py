# Generic libraries imports
import random
import csv
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from datetime import datetime
import tracemalloc
import multiprocessing

from qiskit.transpiler.passes.analysis import depth

# Local application/library imports
import src.backend_gen as backendGen
import src.circuit_gen as circuitGen
import src.qubit_mapping as qMap


def generateResults():
    tracemalloc.start()
    iteration = 25
    timeout_duration = 60  # Maximum allowed time per result generation in seconds

    for i in range(iteration):
        start_time = time.time()

        try:
            # Create a separate process to execute generateOneResult
            process = multiprocessing.Process(target=generateOneResult, args=(i,))
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
def generateOneResult(iterationNumber):
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
    general_info = pd.concat([general_info, pd.DataFrame([pd.Series(dtype='object')])], ignore_index=True)  # just creating a row and filling with NaN
    general_info.at[0, 'date_of_execution'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    our_performance = pd.DataFrame(columns=[
        'depth',
        'total_qubits',
        'gate_count',
        'swap_count',
        't_count',
        't_depth'
    ])
    qiskit_performance = pd.DataFrame(columns=[
        'qiskit_depth',
        'qiskit_total_qubits',
        'qiskit_gate_count',
        'qiskit_swap_count',
        'qiskit_t_count',
        'qiskit_t_depth'
    ])

    workingDir = 'result' + str(iterationNumber)

    general_info.at[0, 'name'] = workingDir

    workingDir = 'results/' + workingDir
    os.makedirs(workingDir, exist_ok=True)

    print('Generating ' + workingDir + ':')

    # Parameters
    num_modules = 7
    module_max_qubits = 4
    module_max_gates = 10
    reduced_distance = None
    max_allowed_weight = 5
    num_qubits_x = 100
    num_qubits_y = 100
    heuristic = False
    save_backend = False
    seed = random.randint(1, int(1e4))

    # %% 3 # failing_seed result file creation and Seed saving
    general_info.at[0, 'seed'] = seed

    failing_seed_path = os.path.join(workingDir, 'failing_seed.txt')
    with open(failing_seed_path, 'w') as file:
        file.write(str(seed) + '\n')

    # %% 4 # Random circuit generation:
    start_time = time.time()
    circuit = circuitGen.RandomCircuit(num_modules, module_max_qubits, module_max_gates,
                                       seed)  # defining circuit properties
    circuit.gen_random_circuit()  # actually generating the circuit with the properties specified above
    end_time = time.time()
    total_time = end_time - start_time

    # %% 5 # Dependency graph generation + saving:
    '''total_time += circuit.generate_dependency_graph(
        workingDir)  # here we specify where we want the png file to be saved //todo in giorgio file this wasn't generated'''

    # %% 6 # Generating the quantum circuit mapping
    ourBackend = backendGen.generate_regular_backend(num_qubits_x, num_qubits_y) #backend generation time shouldn't be considered
    start_time = time.time()
    q_map = qMap.QubitMapping(  # defining mapping properties and constrains
        circuit,
        backend=ourBackend,
        coupling_map_dims=(num_qubits_x, num_qubits_y),
        reduced_distance=reduced_distance,
        max_allowed_weight=max_allowed_weight,
        heuristic=heuristic
    )
    q_map.generate_ASAP_qubit_mapping()  # actually generating the mapping with the properties specified above using an ASAP approach
    end_time = time.time()
    total_time += (end_time - start_time)

    # %% 7 # Retrieving Qiskit metrics
    qiskit_metrics = circuit.get_benchmark_metrics(
        backend=q_map.backend,
        coupling_map=q_map.backend.coupling_map
    )
    qiskit_metrics_prefixed = {f"qiskit_{k}": v for k, v in qiskit_metrics.items()} # just adding "qiskit" prefix on the key
    qiskit_performance = pd.concat([qiskit_performance, pd.DataFrame([qiskit_metrics_prefixed])], ignore_index=True) # saving the metrics into our dataset

    # with open(os.path.join(workingDir, 'metricsQiskit.csv'), 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(qiskit_metrics.keys())  # writing dictionary keys
    #     writer.writerow(qiskit_metrics.values())  # writing the results

    # %% 8 # Saving the quantum circuit mapping results

    # first we just save the metrics in a tabular format
    # with open(os.path.join(workingDir, 'metrics.csv'), 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(q_map.benchmark_metrics.keys())  # writing dictionary keys
    #     writer.writerow(q_map.benchmark_metrics.values())  # writing the results
    # then we save them in a more human readable form


    metrics_content = [
        f"Total time: {total_time}\n",
        f"Seed: {seed}\n",
        'Fixed Parameters:\n'
        f"Number of modules:  {num_modules}\n",
        f"Number of max qubits per module:  {module_max_qubits}\n",
        f"Number of max gates per module:  {module_max_gates}\n",
        f"Reduced distance:  {reduced_distance}\n",
        f"Maximum allowed weight:  {max_allowed_weight}\n",
        f"Number of qubits x:  {num_qubits_x}\n",
        f"Number of qubits y:  {num_qubits_y}\n",
        f"Heuristic:  {heuristic}\n",
        f"Qubit modules:  {q_map.modules_qubits}\n",
        f"Qubit mapping: {q_map.qubit_mapping}\n",
        'Metrics:\n'
    ]
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

    # our metrics
    our_metrics = q_map.benchmark_metrics
    our_performance = pd.concat([our_performance, pd.DataFrame([our_metrics])], ignore_index=True)  # saving the metrics into our dataset

    # for key, value in q_map.benchmark_metrics.items():
    #     metrics_content.append(f'{key}: {value}\n')
    #
    # with open(os.path.join(workingDir, 'metrics.txt'), 'w') as file:
    #     file.writelines(metrics_content)



    # now we need to save everything on disk:

    # unifying dataframes on the same row
    combined_data = pd.concat([general_info, our_performance, qiskit_performance], axis=1)
        # csv format (useful for importing into spreadsheets softwares)
    combined_data.to_csv(os.path.join(workingDir, 'metrics.csv'), index=False)
        # txt format (human readable)
    with open(os.path.join(workingDir, 'metrics.txt'), 'w') as file:
        file.write("### Results: ###\n")
        file.write(combined_data.to_string(index=False))

    # if everything worked correctly we can delete the failing_seed file. if the process is killed or throws an execption
    #   the file remains and we can see the seed later
    if os.path.exists(failing_seed_path):
        os.remove(failing_seed_path)
        print(f"\033[92m{workingDir} generated successfully\033[0m")
    else:
        print(f"\033[91mUnexpected error in failing_seed removal\033[0m")



if __name__ == '__main__':
    generateResults()
