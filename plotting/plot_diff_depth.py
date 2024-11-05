import pandas as pd
import matplotlib.pyplot as plt

diff_depth = {}
opt_lvl = 3
for num_modules in range(4,8):
    path = f'./results/optlvl{opt_lvl}_nm{num_modules}_mmq4_mmg6_rdNone_maw5_nqx100_nqy100_hFalse/total_metrics.csv'
    df   = pd.read_csv(path)

    depth = df['depth']
    qiskit_depth = df['qiskit_depth']

    diff = (qiskit_depth - depth)
    diff_depth[num_modules] = diff.mean()

modules = list(diff_depth.keys())
differences = list(diff_depth.values())

# Crea il bar plot
plt.figure(figsize=(10, 6))
plt.bar(modules, differences, color='skyblue')
plt.xlabel('Number of Modules in Circuit')
plt.ylabel('Average Depth Difference')
plt.title('Average Depth Difference: Qiskit vs Proposed Algorithm')
plt.savefig(f'./plots/depth_optlvl{opt_lvl}.png')