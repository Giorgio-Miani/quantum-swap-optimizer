# Generic libraries imports
import random
import csv
import networkx as nx
import matplotlib.pyplot as plt
import time
import os


# Local application/library imports
import src.backend_gen as backendGen
import src.circuit_gen as circuitGen
import src.qubit_mapping as qMap

def generateResults():
    iteration = 15
    for i in range(iteration):
        try:
            generateOneResult(i)
        except Exception as e:
            print(f"Error at interaction {i}: {e}")
def generateOneResult(iterationNumber):
    workingDir = 'result' + str(iterationNumber)
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

    # %% 3 # Text result file creation and Seed saving
    with open(os.path.join(workingDir, 'metrics.txt'), 'w') as file:
        file.write(str(seed) + '\n')

    # %% 4 # Random circuit generation:
    start_time = time.time()
    circuit = circuitGen.RandomCircuit(num_modules, module_max_qubits, module_max_gates,
                                       seed)  # defining circuit properties
    circuit.gen_random_circuit()  # actually generating the circuit with the properties specified above
    end_time = time.time()
    total_time = end_time - start_time

    # %% 5 # Dependency graph generation + saving:
    total_time += circuit.generate_dependency_graph(
        workingDir)  # here we specify where we want the png file to be saved

    # %% 6 # Generating the quantum circuit mapping
    start_time = time.time()
    q_map = qMap.QubitMapping(  # defining mapping properties and constrains
        circuit,
        backend=backendGen.generate_regular_backend(num_qubits_x, num_qubits_y),
        coupling_map_dims=(num_qubits_x, num_qubits_y),
        reduced_distance=reduced_distance,
        max_allowed_weight=max_allowed_weight,
        heuristic=heuristic
    )
    q_map.generate_ASAP_qubit_mapping()  # actually generating the mapping with the properties specified above using an ASAP approach
    end_time = time.time()
    total_time += (end_time - start_time)

    # %% 7 # Saving the quantum circuit mapping results

    # first we just save the metrics in a tabular format
    with open(os.path.join(workingDir, 'metrics.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(q_map.benchmark_metrics.keys())  # writing dictionary keys
        writer.writerow(q_map.benchmark_metrics.values())  # writing the results

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
    for key, value in q_map.benchmark_metrics.items():
        metrics_content.append(f'{key}: {value}\n')

    with open(os.path.join(workingDir, 'metrics.txt'), 'w') as file:
        file.writelines(metrics_content)

    print(workingDir + ' generated successfully\n')

if __name__ == '__main__':
    generateResults()
