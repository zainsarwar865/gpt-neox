import numpy as np
import matplotlib.pyplot as plt

# timings in ms
timers = [0.283, 0.431, 0.211, 0.154*2, 0.187]  
#labels = ["Gather", "Copy (D→H)", "Grouped GEMM", "Scatter"]
labels = ["Router", "Binning", "Gather", "Grouped GEMM", "Scatter"]

# normalize to percentages
total = sum(timers)
percents = [t / total * 100 for t in timers]
for i in range(len(timers)):
    print(labels[i], percents[i])
# --- donut chart ---
fig, ax = plt.subplots(figsize=(6,6))
wedges, texts, autotexts = ax.pie(
    percents,
    labels=labels,
    autopct=lambda p: f"{p:.1f}%",
    startangle=140,
    pctdistance=0.8,
    wedgeprops=dict(width=0.4, edgecolor="w"),
    textprops=dict(size=12)
)

ax.set_title("Breakdown of StructMoE Overhead (Relative %)", fontsize=14, pad=20)
plt.savefig("struct_overhead_donut.pdf", dpi=300)


# --- horizontal bar chart (alternative) ---
fig, ax = plt.subplots(figsize=(7,4))
bars = ax.barh(labels, percents, color=plt.cm.viridis([0.2, 0.5, 0.7, 0.9]))
ax.set_xlabel("Percent of total StructMoE overhead (%)")
ax.set_xlim(0, 100)

# annotate bars
for bar, pct in zip(bars, percents):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f"{pct:.1f}%", va='center', fontsize=11)

ax.set_title("Breakdown of StructMoE Overhead (Relative %)", fontsize=14, pad=10)
plt.tight_layout()
plt.savefig("struct_overhead.pdf", dpi=300)