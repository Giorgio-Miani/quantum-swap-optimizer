import pandas as pd
import matplotlib.pyplot as plt

swaps = {}
opt_lvl = 3
for num_modules in range(4,8):
    path = f'./results/optlvl{opt_lvl}_nm{num_modules}_mmq4_mmg6_rdNone_maw5_nqx100_nqy100_hFalse/total_metrics.csv'
    df   = pd.read_csv(path)

    swaps[num_modules] = df['swap_count'].mean()

modules = list(swaps.keys())
differences = list(swaps.values())

# Crea il bar plot
plt.figure(figsize=(10, 6))
plt.bar(modules, differences, color='skyblue')
plt.xlabel('Number of Modules in Circuit')
plt.ylabel('Average number of SWAPS')
plt.title('Average number of SWAPS used')
plt.savefig(f'./plots/swaps_avg_optlvl{opt_lvl}.png')