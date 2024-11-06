import pandas as pd
import matplotlib.pyplot as plt

diff_gate_count = {}
opt_lvl = 3
for num_modules in range(4,8):
    path = f'./results/optlvl{opt_lvl}_nm{num_modules}_mmq4_mmg6_rdNone_maw5_nqx100_nqy100_hFalse/total_metrics.csv'
    df   = pd.read_csv(path)

    gate_count = df['gate_count']
    qiskit_gate_count = df['qiskit_gate_count']

    diff = (qiskit_gate_count - gate_count)
    diff_gate_count[num_modules] = diff.mean()

modules = list(diff_gate_count.keys())
differences = list(diff_gate_count.values())

# Crea il bar plot
plt.figure(figsize=(10, 6))
plt.bar(modules, differences, color='skyblue')
plt.xlabel('Number of Modules in Circuit')
plt.ylabel('Average Gate Count Difference')
plt.title('Average Gate Count Difference: Qiskit vs Proposed Algorithm')
plt.savefig(f'./plots/gate_count_optlvl{opt_lvl}.png')