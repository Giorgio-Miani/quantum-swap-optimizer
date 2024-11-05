import pandas as pd
import matplotlib.pyplot as plt

diff_t_count = {}
opt_lvl = 3
for num_modules in range(4,8):
    path = f'./results/optlvl{opt_lvl}_nm{num_modules}_mmq4_mmg6_rdNone_maw5_nqx100_nqy100_hFalse/total_metrics.csv'
    df   = pd.read_csv(path)

    t_count = df['t_count']
    qiskit_t_count = df['qiskit_t_count']

    diff = (qiskit_t_count - t_count)
    diff_t_count[num_modules] = diff.mean()

modules = list(diff_t_count.keys())
differences = list(diff_t_count.values())

# Crea il bar plot
plt.figure(figsize=(10, 6))
plt.bar(modules, differences, color='skyblue')
plt.xlabel('Number of Modules in Circuit')
plt.ylabel('Average T-Count Difference')
plt.title('Average T-Count Difference: Qiskit vs Proposed Algorithm')
plt.savefig(f'./plots/Tcount_optlvl{opt_lvl}.png')