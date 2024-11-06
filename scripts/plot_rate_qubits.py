import os
import sys
# Manually specify the path to the 'src' directory
src_path = os.path.abspath(os.path.join(os.getcwd(), './src'))

# Add the specified path to the system path
sys.path.append(src_path)

import pandas as pd
import matplotlib.pyplot as plt
import circuit_gen as circuitGen

avg_perc_qubits = {}
opt_lvl = 3
for num_modules in range(4,8):
    path = f'./results/optlvl{opt_lvl}_nm{num_modules}_mmq4_mmg6_rdNone_maw5_nqx100_nqy100_hFalse/total_metrics.csv'
    df   = pd.read_csv(path)
    
    perc_qubits = 0
    num = 0
    for index, row in df.iterrows():
        num += 1
        seed = row['seed']
        qubits_occupied = row['total_qubits']

        circuit = circuitGen.RandomCircuit(num_modules, 4, 6, seed)
        circuit.gen_random_circuit()
        circuit_qubits = circuit.qubit_counter
        perc_qubits += circuit_qubits/qubits_occupied
    avg_perc_qubits[num_modules] = perc_qubits/(index+1)


modules = list(avg_perc_qubits.keys())
differences = list(avg_perc_qubits.values())

# Crea il bar plot
plt.figure(figsize=(10, 6))
plt.bar(modules, differences, color='skyblue')
plt.xlabel('Number of Modules in Circuit')
plt.ylabel('Average Rate')
plt.title('Average Rate of logical Qubits and physical Qubits occupied')
plt.savefig(f'./plots/rate_qubits_optlvl{opt_lvl}.png')