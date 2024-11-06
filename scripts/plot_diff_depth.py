import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

Diff_depth = {'Num_modules':[], 'Diff_depth':[], 'Opt_lvl':[]}
for opt_lvl in range(1,4):
    for num_modules in range(4,8):
        path = f'./results/optlvl{opt_lvl}_nm{num_modules}_mmq4_mmg6_rdNone_maw5_nqx100_nqy100_hFalse/total_metrics.csv'
        if not os.path.exists(path):
            continue
        
        df   = pd.read_csv(path)

        depth = df['depth']
        qiskit_depth = df['qiskit_depth']

        diff = (qiskit_depth - depth)
        Diff_depth['Num_modules'].append(num_modules)
        Diff_depth['Diff_depth'].append(diff.mean())
        Diff_depth['Opt_lvl'].append(opt_lvl)

df = pd.DataFrame(Diff_depth)


# Crea il bar plot
palette = ['lightblue','violet','pink']
plt.figure(figsize=(10, 6))
sns.barplot(x='Num_modules', y='Diff_depth', hue='Opt_lvl', data=df, palette=palette)
plt.xlabel('Number of Modules in Circuit')
plt.ylabel('Average Number of Qubits Difference')
plt.title('Average Number of Qubits Difference: Qiskit vs Proposed Algorithm')
plt.savefig(f'./plots/diff_depth.png')