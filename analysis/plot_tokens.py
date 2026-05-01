
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
from scipy.ndimage import gaussian_filter1d


DUMP_DIR = "/home/zsarwar/Projects/CODE/Code/gpt_neox_old/c2_configs/bash_scripts/tokens_per_lora_dumps/"
OUTPUT_DIR = os.path.join(DUMP_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_step(filename):
    match = re.search(r"step_(\d+)", filename)
    return int(match.group(1)) if match else -1


def load_and_normalize(filepath):
    data = torch.load(filepath)
    normalized = {}
    for layer_idx, token_lists in data.items():
        token_tensor = torch.stack(token_lists)  # [batch, num_loras]
        total_tokens = token_tensor.sum(dim=-1, keepdim=True)
        total_tokens[total_tokens == 0] = 1
        percentage = token_tensor / total_tokens
        mean_percentage = percentage.mean(dim=0)
        normalized[layer_idx] = mean_percentage.numpy()
    return normalized


def smooth_matrix(y_values, smooth_sigma):
    if smooth_sigma > 0:
        # Apply Gaussian smoothing along the layer (row) axis
        return gaussian_filter1d(y_values, sigma=smooth_sigma, axis=0)
    else:
        return y_values


def plot_distribution(normalized, step, smooth_sigma):
    sorted_layers = sorted(normalized.keys())
    lora_count = len(next(iter(normalized.values())))
    x = np.array(sorted_layers)
    y_values = np.stack([normalized[layer] for layer in sorted_layers], axis=0)  # [layers, loras]

    # Apply smoothing
    y_values = smooth_matrix(y_values, smooth_sigma)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative = np.zeros_like(x, dtype=np.float32)

    # --- 1. Generate unique colors ---
    # 'gist_rainbow' or 'turbo' are good for a high number of SMs
    # 'tab20' is great if you have <= 20 SMs

    
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, lora_count))
    bold_colors = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Dark2").colors[:6])



    for i in range(lora_count):
        y = y_values[:, i]
        ax.fill_between(x, cumulative, cumulative + y, label=f"SM {i}", alpha=0.7, color=bold_colors[i])
        cumulative += y

    title_font = {"fontsize": 16}
    label_font = {"fontsize": 14}

    #ax.set_title(f"Layerwise Token-LoRE Routing Distribution", fontdict=title_font)
    ax.set_xlabel("Layer Index", fontdict=label_font)
    ax.set_ylabel("Fraction of Tokens Seen by each SM",fontdict=label_font)
    ax.set_ylim(0, 1)
    ax.set_xlim(x[0], x[-1])
    # ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0), fontsize="small", ncol=2)
    ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize="small",
    ncol=1,
    borderaxespad=0.0,
    frameon=False,
    )
    plt.tight_layout()
    return fig


def main(smooth_sigma):
    dump_files = [
        f for f in os.listdir(DUMP_DIR)
        if re.match(r"step_\d+\.pt", f)
    ]
    dump_files = sorted(dump_files, key=extract_step)

    if not dump_files:
        print("No dump files found.")
        return

    for fname in dump_files:
        step = extract_step(fname)
        fpath = os.path.join(DUMP_DIR, fname)
        normalized_data = load_and_normalize(fpath)
        fig = plot_distribution(normalized_data, step, smooth_sigma)
        out_path = os.path.join(OUTPUT_DIR, f"distribution_step_{step}.pdf")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", type=float, default=1.0,
                        help="Smoothing sigma for Gaussian blur (e.g., 1.0 or 2.0). 0 disables smoothing.")
    args = parser.parse_args()
    main(smooth_sigma=args.smooth)