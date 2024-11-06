import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

diff_qubits = {'Num_modules':[], 'TCount':[], 'Opt_lvl':[]}
for opt_lvl in range(1,4):
    for num_modules in range(4,8):
        path = f'./results/optlvl{opt_lvl}_nm{num_modules}_mmq4_mmg6_rdNone_maw5_nqx100_nqy100_hFalse/total_metrics.csv'
        if not os.path.exists(path):
            continue
        
        df   = pd.read_csv(path)

        t_count = df['t_count']
        qiskit_t_count = df['qiskit_t_count']

        diff = (qiskit_t_count - t_count)
        diff_qubits['Num_modules'].append(num_modules)
        diff_qubits['TCount'].append(diff.mean())
        diff_qubits['Opt_lvl'].append(opt_lvl)

df = pd.DataFrame(diff_qubits)


# Crea il bar plot
palette = ['lightblue','violet','pink']
plt.figure(figsize=(10, 6))
sns.barplot(x='Num_modules', y='TCount', hue='Opt_lvl', data=df, palette=palette)
plt.xlabel('Number of Modules in Circuit')
plt.ylabel('Average Number of Qubits Difference')
plt.title('Average Number of Qubits Difference: Qiskit vs Proposed Algorithm')
plt.savefig(f'./plots/diff_Tcount.png')