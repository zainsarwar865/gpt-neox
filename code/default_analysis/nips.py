import numpy as np
import matplotlib.pyplot as plt

# 1) Manually enter your data here:
hidden_dims     = [0.57, 1.96, 7.3]              # example dims
baseline_perps  = [32.95, 23.77, 20.47]             # replace with your final baseline perps
method_perps    = [31.98, 23.3, 20.17]             # replace with your final Default MoE perps

# 2) Prepare x positions so they’re evenly spaced
x = np.arange(len(hidden_dims))  # [0, 1, 2, ...]

# 3) Plot
plt.rcParams['figure.dpi'] = 300
plt.figure(figsize=(4,2.5))

# baseline line with smaller markers
plt.plot(x,
         baseline_perps,
         marker='o',
         linestyle='-',
         linewidth=1.0,
         markersize=3,            # smaller marker size
         label='TopK')

# method line with smaller markers
plt.plot(x,
         method_perps,
         marker='s',
         linestyle='-',
         linewidth=1.0,
         markersize=3,            # smaller marker size
         label='Default MoE')

# 4) Labels, ticks, legend
plt.xlabel('Total parameters (Billion)')
plt.ylabel('Perplexity')
plt.xticks(x, hidden_dims)   # show 512, 1024, 2048 at positions 0,1,2
plt.grid(True, linestyle=':', linewidth=0.5)
plt.legend(loc='best', fontsize=8)

# 5) Optional: pad y‐limits by 5%
all_perps = np.array(baseline_perps + method_perps)
y_min, y_max = all_perps.min(), all_perps.max()
y_pad = (y_max - y_min) * 0.05
plt.ylim(y_min - y_pad, y_max + y_pad)

# 6) Save & show
plt.tight_layout()
plt.savefig('final_perplexity_hidden_dims_ablation.pdf', bbox_inches='tight')
plt.show()




# # 1) Manually enter your data here:
# hidden_dims     = [3e-4, 5e-4, 7e-4, 9e-4]              # example dims
# baseline_perps  = [23.05, 21.82, 21.13, 22.45]             # replace with your final baseline perps
# method_perps    = [22.62, 21.78, 21.21, 21.14]             # replace with your final Default MoE perps

# # 2) Prepare x positions so they’re evenly spaced
# x = np.arange(len(hidden_dims))  # [0, 1, 2, ...]

# # 3) Plot
# plt.rcParams['figure.dpi'] = 300
# plt.figure(figsize=(4,2.5))

# # baseline line with smaller markers
# plt.plot(x,
#          baseline_perps,
#          marker='o',
#          linestyle='-',
#          linewidth=1.0,
#          markersize=3,            # smaller marker size
#          label='TopK')

# # method line with smaller markers
# plt.plot(x,
#          method_perps,
#          marker='s',
#          linestyle='-',
#          linewidth=1.0,
#          markersize=3,            # smaller marker size
#          label='Default MoE')

# # 4) Labels, ticks, legend
# plt.xlabel('Learning Rate')
# plt.ylabel('Perplexity')
# sci_labels = [f"{lr:.0e}" for lr in hidden_dims]
# plt.xticks(x, sci_labels)
# # plt.xticks(x, hidden_dims)   # show 512, 1024, 2048 at positions 0,1,2
# plt.grid(True, linestyle=':', linewidth=0.5)
# plt.legend(loc='best', fontsize=8)

# # 5) Optional: pad y‐limits by 5%
# all_perps = np.array(baseline_perps + method_perps)
# y_min, y_max = all_perps.min(), all_perps.max()
# y_pad = (y_max - y_min) * 0.05
# plt.ylim(y_min - y_pad, y_max + y_pad)

# # 6) Save & show
# plt.tight_layout()
# plt.savefig('final_perplexity_lr_sweep_ablation.pdf', bbox_inches='tight')
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# # Data
# configs = ["32c4", "32c2", "32c1", "8c2", "8c1"]
# default_vector = [21.54, 22.26, 22.95, 21.94, 22.67]
# topk           = [22.06, 22.32, 22.96, 22.11, 23.07]

# # 1) Introduce a spacing factor >1
# spacing = 1.2
# x = np.arange(len(configs)) * spacing

# # 2) Keep bars nice and fat
# width = 0.55

# plt.rcParams['figure.dpi'] = 300
# plt.figure(figsize=(4, 2.2))

# # 3) Plot bars at the new x positions
# plt.bar(x - width/2, default_vector, width,
#         label="Default Vector", edgecolor='black')
# plt.bar(x + width/2, topk,           width,
#         label="TopK",           edgecolor='black')

# # 4) Labels & ticks
# plt.xlabel("Sparsity Configuration", fontsize=8)
# plt.ylabel("Final Perplexity", fontsize=8)
# plt.xticks(x, configs, fontsize=8)
# plt.yticks(fontsize=9)
# plt.legend(fontsize=9)

# # 5) Y‐axis range
# y_min = 20
# y_max = max(default_vector + topk) * 1.05
# plt.ylim(y_min, y_max)

# # 6) Adjust x‐limits to fit the stretched x
# plt.xlim(x[0] - spacing/2, x[-1] + spacing/2)

# # 7) Annotate
# for xi, dv, tk in zip(x, default_vector, topk):
#     plt.text(xi - width/2, dv + (y_max - y_min)*0.02,
#              f"{dv:.2f}", ha='center', va='bottom', fontsize=6.5, fontweight='bold')
#     plt.text(xi + width/2, tk + (y_max - y_min)*0.02,
#              f"{tk:.2f}", ha='center', va='bottom', fontsize=6.5, fontweight='bold')

# plt.tight_layout(pad=0.2)
# plt.savefig("final_perplexity_bar_chart.pdf", bbox_inches="tight")
# plt.show()













# configs = ["32c4", "32c2", "32c1", "8c2", "8c1"]
# default_vector = [21.54, 22.26, 22.95, 21.94, 22.67]
# topk           = [22.06, 22.32, 22.96, 22.11, 23.07]

# from fractions import Fraction
# # 1) Manually enter your data here:
# hidden_dims     = [1/4, 1/8, 1/16, 1/32, 1/64]              # example dims
# baseline_perps  = [22.11, 22.06,  22.32, 22.96, 16.91]             # replace with your final baseline perps
# method_perps    = [21.94, 21.54, 22.26, 22.95, 16.66]             # replace with your final Default MoE perps

# # 2) x positions
# x = np.arange(len(hidden_dims))

# # 3) Plot
# plt.rcParams['figure.dpi'] = 300
# plt.figure(figsize=(4,2.5))
# plt.plot(x, baseline_perps, marker='o', linestyle='-', linewidth=1, markersize=3, label='TopK')
# plt.plot(x, method_perps,   marker='s', linestyle='-', linewidth=1, markersize=3, label='Default MoE')

# # 4) Labels & custom fraction ticks
# plt.xlabel('MoE Sparsity Factor')
# plt.ylabel('Perplexity')

# # convert floats back to fraction strings
# frac_labels = [str(Fraction(h).limit_denominator()) for h in hidden_dims]
# plt.xticks(x, frac_labels, fontsize=8)

# plt.grid(True, linestyle=':', linewidth=0.5)
# plt.legend(loc='best', fontsize=8)

# # 5) Y‐limits padding
# all_p = np.array(baseline_perps + method_perps)
# y_min, y_max = all_p.min(), all_p.max()
# pad = (y_max - y_min) * 0.05
# plt.ylim(y_min - pad, y_max + pad)

# # 6) Save
# plt.tight_layout()
# plt.savefig('sparsity_factor_ablation.pdf', bbox_inches='tight')
# plt.show()