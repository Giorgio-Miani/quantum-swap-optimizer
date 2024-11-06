import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

diff_qubits = {'Num_modules':[], 'Diff_qubits':[], 'Opt_lvl':[]}
for opt_lvl in range(1,4):
    for num_modules in range(4,8):
        path = f'./results/optlvl{opt_lvl}_nm{num_modules}_mmq4_mmg6_rdNone_maw5_nqx100_nqy100_hFalse/total_metrics.csv'
        if not os.path.exists(path):
            continue
        
        df   = pd.read_csv(path)

        total_qubits = df['total_qubits']
        qiskit_total_qubits = df['qiskit_total_qubits']

        diff = (qiskit_total_qubits - total_qubits)
        diff_qubits['Num_modules'].append(num_modules)
        diff_qubits['Diff_qubits'].append(diff.mean())
        diff_qubits['Opt_lvl'].append(opt_lvl)

df = pd.DataFrame(diff_qubits)


# Crea il bar plot
palette = ['lightblue', 'violet', 'pink']
plt.figure(figsize=(10, 6))
sns.barplot(x='Num_modules', y='Diff_qubits', hue='Opt_lvl', data=df, palette=palette)
plt.xlabel('Number of Modules in Quantum Circuit', fontsize=14)
plt.ylabel('Average Number of Qubits Difference', fontsize=14)
plt.title('Average Number of Qubits Difference: Qiskit vs Proposed Algorithm', fontsize=16)
plt.legend(title='Optimization Level', title_fontsize='13', fontsize=11, loc='upper left')
plt.savefig(f'./plots/diff_qubits.png')