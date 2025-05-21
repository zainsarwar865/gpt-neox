import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import re
import itertools


def load_data(csv_path, usecols, rename_cols, label_map=None, col_order=None):

    df = pd.read_csv(csv_path, usecols=usecols)
    df.columns = rename_cols
    if label_map:
        df.rename(columns=label_map, inplace=True)
    for col in df.columns[1:]: 
        df[col + "_Perplexity"] = np.exp(df[col])


    # df = df[df["Step"] > 1000]
    batch_size = 2**21

    df["Tokens"] = df["Step"] * batch_size
    updated_labels = list(df.columns[1:len(rename_cols)])
    if col_order:
        updated_labels = [label for label in col_order if label in updated_labels] 
    return df, updated_labels


# def plot_perplexity(df, labels, span=1000, name="plot.pdf"):

#     plt.rcParams['figure.dpi'] = 300
#     fontdict = {'family': 'serif', 'weight': 'normal', 'size': 10}
#     plt.rc('font', **fontdict)

#     plt.clf() 
#     plt.figure(figsize=(4, 2.5))

#     for label in labels:
#         smoothed_col = label + "_EWM"
#         df[smoothed_col] = df[label + "_Perplexity"].ewm(span=span).mean()
#         plt.plot(df["Tokens"], df[smoothed_col], label=label, linestyle="solid")

#     plt.xlabel("Tokens")
#     plt.legend(loc="lower left")
#     plt.grid(True)

#     ax = plt.gca()

#     ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
#     ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e9:.0f}B'))


#     perplexity_columns = [col for col in df.columns if col.endswith("_EWM")]


#     y_min = df[perplexity_columns].min().min()
#     plt.ylim(y_min * 0.97, 32)  # NEED 42 FOR 512


#     plt.savefig(name, format="pdf", bbox_inches="tight")
#     plt.show()

def plot_perplexity(df, labels, span=1000, name="plot.pdf", y_threshold=32):
    import matplotlib.ticker as mticker
    import numpy as np

    plt.rcParams['figure.dpi'] = 300
    fontdict = {'family': 'serif', 'weight': 'normal', 'size': 10}
    plt.rc('font', **fontdict)

    # drop zero-step rows & recompute tokens
    df = df[df["Step"] > 0].copy()
    batch_size = 2**21
    df["Tokens"] = df["Step"] * batch_size

    # compute all EWM curves up front
    for label in labels:
        df[label + "_EWM"] = df[label + "_Perplexity"].ewm(span=span).mean()

    plt.clf()
    plt.figure(figsize=(4, 2.5))
    for label in labels:
        print(label, (df[label + "_EWM"].iloc[-1]))
        plt.plot(df["Tokens"], df[label + "_EWM"], label=label, linewidth=0.75)
    plt.xlabel("Tokens")
    plt.grid(True)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x/1e9:.0f}B')
    )

    # clamp y-axis
    y_min = min(df[label + "_EWM"].min() for label in labels)
    plt.ylim(y_min * 0.97, y_threshold)

    # find for each label the tokens where EWM <= threshold
    starts, ends = [], []
    for label in labels:
        col = label + "_EWM"
        under = df[df[col] <= y_threshold]["Tokens"]
        if not under.empty:
            starts.append(under.min())
            ends.append(under.max())

    if starts and ends:
        # intersection: show only where ALL curves are <= threshold
        x_start = max(starts)
        x_end   = min(ends)
        ax.set_xlim(x_start, x_end)

    plt.legend(loc="lower left")
    plt.savefig(name, format="pdf", bbox_inches="tight")
    plt.show()

def extract_lr(label):
    match = re.search(r"\d+e-\d+", label)
    return match.group(0) if match else "Unknown"

def extract_category(label):
    return label.split(extract_lr(label))[0].strip() if "e-" in label else label

# def plot_perplexity_category(df, labels, span=1000, name="plot.pdf"):
#     plt.rcParams['figure.dpi'] = 300
#     fontdict = {'family': 'serif', 'weight': 'normal', 'size': 10}
#     plt.rc('font', **fontdict)
#     category_styles = {}
#     line_styles = ["solid", "dashed", "dotted", "dashdot"] 
#     style_cycle = itertools.cycle(line_styles) 

#     for label in labels:
#         category = extract_category(label)
#         if category not in category_styles:
#             category_styles[category] = next(style_cycle)
#     color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
#     lr_colors = {}
#     plt.figure(figsize=(4, 2.5))


#     for label in labels:
#         smoothed_col = label + "_EWM"
#         df[smoothed_col] = df[label + "_Perplexity"].ewm(span=span).mean()

#         category = extract_category(label)
#         lr = extract_lr(label)

#         if lr not in lr_colors:
#             lr_colors[lr] = next(color_cycle)

#         plt.plot(df["Tokens"], df[smoothed_col], label=label, linestyle=category_styles[category], color=lr_colors[lr], lw=0.75)

#     plt.xlabel("Tokens")
#     plt.grid(True)
#     ax = plt.gca()
#     ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
#     ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e9:.0f}B'))

#     perplexity_columns = [col for col in df.columns if col.endswith("_EWM")]

#     y_min = df[perplexity_columns].min().min()
#     plt.ylim(y_min * 0.97, 32) 

#     category_legend = [plt.Line2D([0], [0], color="black", linestyle=category_styles[cat], lw=2, label=cat) for cat in category_styles]
#     lr_legend = [plt.Line2D([0], [0], color=lr_colors[lr], linestyle="solid", lw=2, label=lr) for lr in lr_colors]
#     plt.legend(handles=category_legend + lr_legend, loc="upper right", ncol=2)

#     plt.savefig(name, format="pdf", bbox_inches="tight")
#     plt.show()



import itertools
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_perplexity_category(df, labels, span=1000, y_threshold=32, name="plot.pdf"):
    plt.rcParams['figure.dpi'] = 300
    plt.rc('font', family='serif', weight='normal', size=10)

    # 1) drop Step==0 & recompute Tokens
    df = df[df["Step"] > 0].copy()
    df["Tokens"] = df["Step"] * (2**21)

    # 2) compute smoothed columns
    for label in labels:
        df[label + "_EWM"] = df[label + "_Perplexity"].ewm(span=span).mean()

    # 3) crop to when ANY curve first dips <= threshold
    any_below = None
    for label in labels:
        cur = df[label + "_EWM"] <= y_threshold
        any_below = cur if any_below is None else (any_below | cur)
    if any_below.any():
        first_idx   = any_below.idxmax()
        start_token = df.at[first_idx, "Tokens"]
        df = df[df["Tokens"] >= start_token]

    # 4) prepare style maps
    style_cycle    = itertools.cycle(["solid", "dashed", "dotted", "dashdot"])
    category_styles = {}
    for label in labels:
        cat = extract_category(label)
        if cat not in category_styles:
            category_styles[cat] = next(style_cycle)

    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    lr_colors   = {}

    # 5) plot each curve
    plt.figure(figsize=(4,2.5))
    for label in labels:
        cat = extract_category(label)
        lr  = extract_lr(label)
        if lr not in lr_colors:
            lr_colors[lr] = next(color_cycle)
        print(label, df[label + "_EWM"].iloc[-1])
        plt.plot(
            df["Tokens"],
            df[label + "_EWM"],
            label=label,
            linestyle=category_styles[cat],
            color=lr_colors[lr],
            lw=0.75
        )

    # 6) finalize formatting
    plt.xlabel("Tokens")
    plt.grid(True)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x/1e9:.0f}B')
    )
    # y‐limit up to threshold
    min_ew = min(df[label + "_EWM"].min() for label in labels)
    plt.ylim(min_ew * 0.97, y_threshold)

    # build two‐part legend
    category_legend = [
        plt.Line2D([0], [0], color="black",
                   linestyle=category_styles[c],
                   lw=2, label=c)
        for c in category_styles
    ]
    lr_legend = [
        plt.Line2D([0], [0],
                   color=lr_colors[l],
                   linestyle="solid",
                   lw=2, label=l)
        for l in lr_colors
    ]
    plt.legend(handles=category_legend + lr_legend,
               loc="upper right", ncol=2, fontsize=8)

    plt.savefig(name, bbox_inches="tight")
    plt.show()






import re
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def extract_config(label):
    """Grab the '8c1', '32c2', etc. from the end of your label."""
    m = re.search(r'(\d+c\d)$', label)
    return m.group(1) if m else "unknown"

# def plot_perplexity_category(df, labels, span=1000, y_threshold=32, name="plot.pdf"):
#     plt.rcParams['figure.dpi'] = 300
#     plt.rc('font', family='serif', weight='normal', size=10)

#     # 1) drop Step==0 & recompute Tokens
#     df = df[df["Step"] > 0].copy()
#     df["Tokens"] = df["Step"] * (2**21)

#     # 2) compute all EWM columns
#     for label in labels:
#         df[label + "_EWM"] = df[label + "_Perplexity"].ewm(span=span).mean()

#     # 3) crop to first hit of ANY curve ≤ threshold
#     any_below = None
#     for label in labels:
#         below = df[label + "_EWM"] <= y_threshold
#         any_below = below if any_below is None else (any_below | below)
#     if any_below.any():
#         first_idx   = any_below.idxmax()
#         start_token = df.at[first_idx, "Tokens"]
#         df = df[df["Tokens"] >= start_token]

#     # 4) prepare style maps
#     line_styles = ["solid", "dashed", "dotted", "dashdot"]
#     style_cycle = itertools.cycle(line_styles)
#     category_styles = {}
#     for label in labels:
#         cat = "Default MoE" if label.startswith("Default MoE") else "TopK"
#         if cat not in category_styles:
#             category_styles[cat] = next(style_cycle)

#     color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
#     config_colors = {}
#     for label in labels:
#         cfg = extract_config(label)
#         if cfg not in config_colors:
#             config_colors[cfg] = next(color_cycle)

#     # 5) plot
#     plt.figure(figsize=(4, 2.5))
#     for label in labels:
#         cat = "Default MoE" if label.startswith("Default MoE") else "TopK"
#         cfg = extract_config(label)
#         print(label, df[label + "_EWM"].iloc[-1])
#         plt.plot(
#             df["Tokens"],
#             df[label + "_EWM"],
#             label=label,
#             linestyle=category_styles[cat],
#             color=config_colors[cfg],
#             lw=0.75
#         )

#     # 6) formatting
#     plt.xlabel("Tokens")
#     plt.grid(True)
#     ax = plt.gca()
#     ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
#     ax.xaxis.set_major_formatter(
#         mticker.FuncFormatter(lambda x, _: f'{x/1e9:.0f}B')
#     )
#     # y-limit
#     y_min = min(df[label + "_EWM"].min() for label in labels)
#     plt.ylim(y_min * 0.97, y_threshold)

#     # 7) legends: first line‐style by category, then color by config
#     cat_legend = [
#         plt.Line2D([0], [0], color="black",
#                    linestyle=category_styles[cat],
#                    lw=2,
#                    label=cat)
#         for cat in category_styles
#     ]
#     cfg_legend = [
#         plt.Line2D([0], [0], color=config_colors[cfg],
#                    linestyle="solid",
#                    lw=2,
#                    label=cfg)
#         for cfg in config_colors
#     ]
#     plt.legend(handles=cat_legend + cfg_legend,
#                loc="upper right",
#                ncol=2,
#                fontsize=8)

#     plt.savefig(name, bbox_inches="tight")
#     plt.show()















def compare_token_efficiency(df, updated_labels, target_perplexity, batch_size=2**21):
    df["Tokens"] = df["Step"] * batch_size
    token_reach = {}
    for label in updated_labels:
        perplexity_col = label + "_Perplexity" 
        if perplexity_col not in df.columns:
            print(f"...error")
            continue

        threshold_reached = df[df[perplexity_col] <= target_perplexity]
        if not threshold_reached.empty:
            token_value = threshold_reached.iloc[0]["Tokens"] 
            token_reach[label] = token_value
            print(f"{label} reaches perplexity {target_perplexity} at {token_value:.2e}")
        else:
            token_reach[label] = float("inf") 
            print(f"perp not matched..")

    moe_token = token_reach.get("Default MoE", float("inf"))
    topk_token = token_reach.get("TopK", float("inf"))

    if moe_token < float("inf") and topk_token < float("inf"):
        token_difference = topk_token - moe_token
        speedup_factor = topk_token / moe_token
    else:
        token_difference, speedup_factor = None, None 

    print("Token Efficiency:")
    return {
        "Default MoE Tokens": moe_token,
        "Top-K Tokens": topk_token,
        "Token Difference": token_difference,
        "Speedup Factor": speedup_factor
    }








# def extract_config(label):
#     match = re.search(r"\d+c\d+", label)
#     return match.group(0) if match else "Unknown Config"

# def extract_category(label):
#     return label.split(extract_config(label))[0].strip() if "c" in label else label

# def plot_perplexity_category(df, labels, span=1000, name="plot.pdf"):

#     plt.rcParams['figure.dpi'] = 300
#     fontdict = {'family': 'serif', 'weight': 'normal', 'size': 10}
#     plt.rc('font', **fontdict)

#     category_styles = {}
#     line_styles = ["solid", "dashed", "dotted", "dashdot"] 
#     style_cycle = itertools.cycle(line_styles)  


#     for label in labels:
#         category = extract_category(label)
#         if category not in category_styles:
#             category_styles[category] = next(style_cycle)

#     color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
#     config_colors = {}

#     plt.figure(figsize=(4, 2.5))

#     # Apply smoothing and plot
#     for label in labels:
#         smoothed_col = label + "_EWM"
#         df[smoothed_col] = df[label + "_Perplexity"].ewm(span=span).mean()

#         config = extract_config(label)  
#         category = extract_category(label) 

#         if config not in config_colors:
#             config_colors[config] = next(color_cycle)

#         plt.plot(df["Tokens"], df[smoothed_col], label=label, linestyle=category_styles[category], color=config_colors[config], lw=0.75)

#     plt.xlabel("Tokens")
#     plt.grid(True)

#     # Format axes
#     ax = plt.gca()
#     ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
#     ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e9:.0f}B'))
#     perplexity_columns = [col for col in df.columns if col.endswith("_EWM")]
#     y_min = df[perplexity_columns].min().min()
#     plt.ylim(y_min * 0.97, 32)

#     config_legend = [plt.Line2D([0], [0], color=config_colors[conf], linestyle="solid", lw=2, label=conf) for conf in config_colors]
#     category_legend = [plt.Line2D([0], [0], color="black", linestyle=category_styles[cat], lw=2, label=cat) for cat in category_styles]
#     plt.legend(handles=config_legend + category_legend, loc="lower left", ncol=2)
#     plt.savefig(name, format="pdf", bbox_inches="tight")
#     plt.show()



# # EMA ABLATION

# # # Define file path
# csv_path = "/home/zsarwar/Projects/Wandb/data/wandb_export_2025-01-30T20_01_12.323-06_00.csv"
# name = "ema_beta_ablation.pdf"

# # Define columns to load (Step, 1st run loss, 2nd run loss)
# usecols = [0, 1, 7, 4]  
# rename_cols = ["Step", "EMA_Beta_0.999_Loss", "EMA_Beta_0.9_Loss", "EMA_Beta_0.99"]
# # col_order = ["Step", "EMA_Beta_0.999_Loss", "EMA_Beta_0.99", "EMA_Beta_0.9_Loss"]
# # Define new column names
# label_map = {
#     "EMA_Beta_0.999_Loss": "Beta - 0.999",
#     "EMA_Beta_0.9_Loss": "Beta - 0.9",
#     "EMA_Beta_0.99": "Beta - 0.99"
# }

# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols, label_map)
# # Plot data with EWM applied and updated labels
# plot_perplexity(df, labels=updated_labels, span=1000, name=name)



# # SparseMixer ABLATION

# # # Define file path
# # Define file path
# csv_path = "/home/zsarwar/Projects/Wandb/data/wandb_export_2025-01-30T21_15_05.420-06_00_sparsemixer.csv"
# name = "vector_vs_sparsemixer.pdf"

# # Define columns to load (Step, loss for default vector, loss for sparsemixer)
# usecols = [0, 1, 4]  
# rename_cols = ["Step", "Default_Vector_Loss", "SparseMixer_Loss"]

# # Define new column names for cleaner plotting
# label_map = {
#     "Default_Vector_Loss": "Default MoE 8c2",
#     "SparseMixer_Loss": "SparseMixer 8c2"
# }

# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols, label_map)

# # Plot data with EWM applied and updated labels
# plot_perplexity(df, labels=updated_labels, span=1000, name=name)



# # Zero Init ABLATION
# # Define file path
# csv_path = "/home/zsarwar/Projects/Wandb/data/wandb_export_2025-01-30T22_06_21.257-06_00_init.csv"
# name = "zero_init_vs_random_ablation.pdf"

# # Define columns to load (Step, loss for default vector, loss for sparsemixer)
# usecols = [0, 1, 4]  
# rename_cols = ["Step", "zero_init", "random_init"]

# # Define new column names for cleaner plotting
# label_map = {
#     "zero_init": "Zero init",
#     "random_init": "Random init"
# }

# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols, label_map)

# # Plot data with EWM applied and updated labels
# plot_perplexity(df, labels=updated_labels, span=1000, name=name)




# # 2048 DIM
# csv_path = "/home/zsarwar/Projects/gpt_neox/gpt-neox/code/default_analysis/Wandb/2048_hidden_dim.csv"
# name = "hidden_dim_2048_ablation.pdf"

# # Define columns to load (Step, loss for default vector, loss for sparsemixer)
# usecols = [0, 1, 4]  
# rename_cols = ["Step", "TopK - 2048", "Default MoE - 2048"]

# # Define new column names for cleaner plotting
# label_map = {
#     "TopK - 2048": "TopK - 2048",
#     "Default MoE - 2048": "Default MoE - 2048"
# }

# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols, label_map)

# # Plot data with EWM applied and updated labels
# plot_perplexity(df, labels=updated_labels, span=200, name=name)


# # DIM 512
# csv_path = "/home/zsarwar/Projects/gpt_neox/gpt-neox/code/default_analysis/Wandb/512_hidden_dim.csv"
# name = "hidden_dim_512_ablation.pdf"

# # Define columns to load (Step, loss for default vector, loss for sparsemixer)
# usecols = [0, 1, 4]  
# rename_cols = ["Step", "TopK - 512", "Default MoE - 512"]

# # Define new column names for cleaner plotting
# label_map = {
#     "TopK - 512": "TopK - 512",
#     "Default MoE - 512": "Default MoE - 512"
# }

# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols, label_map)
# # Plot data with EWM applied and updated labels
# plot_perplexity(df, labels=updated_labels, span=1000, name=name)



# # DIM 1024
# csv_path = "/home/zsarwar/Projects/gpt_neox/gpt-neox/code/default_analysis/Wandb/1024_hidden_dim.csv"
# name = "hidden_dim_1024_ablation.pdf"

# # Define columns to load (Step, loss for default vector, loss for sparsemixer)
# usecols = [0, 4, 1]  
# rename_cols = ["Step", "Default MoE - 1024", "TopK - 1024"]
# col_order = ["Step", "TopK - 1024", "Default MoE - 1024"]
# # # Define new column names for cleaner plotting
# # label_map = {
# #     "Default MoE - 1024": "Default MoE - 1024",
# #     "TopK - 1024": "TopK - 1024",

# # }

# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols, col_order=col_order)

# # Plot data with EWM applied and updated labels
# plot_perplexity(df, labels=updated_labels, span=1000, name=name)


#LR Sweep

# csv_path = "/home/zsarwar/Projects/gpt_neox/gpt-neox/code/default_analysis/Wandb/lr_sweep.csv"
# name = "lr_sweep_ablation.pdf"

# # Define columns to load (Step, loss for default vector, loss for sparsemixer)
# usecols = [0, 1, 4, 7, 10, 13, 16, 19]  
# rename_cols = ["Step", "Default MoE 9e-4", "Default MoE 7e-4", "TopK 9e-4", "TopK 7e-4", "TopK 5e-4", "Default MoE 3e-4", "TopK 3e-4" ]
# col_order = ["Step", "Default MoE 9e-4", "Default MoE 7e-4", "Default MoE 3e-4", "TopK 9e-4", "TopK 7e-4", "TopK 5e-4", "TopK 3e-4" ]
# # # Define new column names for cleaner plotting
# # label_map = {
# #     "TopK - Dim 1024": "TopK - Dim 1024",
# #     "Default MoE - Dim 1024": "Default MoE - Dim 1024"
# # }

# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols, col_order=col_order)

# # Plot data with EWM applied and updated labels
# plot_perplexity_category(df, labels=updated_labels, span=400, name=name)



csv_path = "/home/zsarwar/Projects/gpt_neox/gpt-neox/code/default_analysis/Wandb/lr_sweep_full.csv"
name = "lr_sweep_ablation_full.pdf"

# Define columns to load (Step, loss for default vector, loss for sparsemixer)
usecols = [0, 1, 4, 7, 10, 13, 16, 19, 22]  

rename_cols = ["Step", "Default MoE 9e-4", "Default MoE 7e-4",  "Default MoE 5e-4", "TopK 9e-4", "TopK 7e-4", "TopK 5e-4", "Default MoE 3e-4", "TopK 3e-4" ]
col_order = ["Step", "Default MoE 9e-4", "Default MoE 7e-4",  "Default MoE 5e-4", "Default MoE 3e-4", "TopK 9e-4", "TopK 7e-4", "TopK 5e-4", "TopK 3e-4" ]
# # Define new column names for cleaner plotting
# label_map = {
#     "TopK - Dim 1024": "TopK - Dim 1024",
#     "Default MoE - Dim 1024": "Default MoE - Dim 1024"
# }

# Load data with new column labels
df, updated_labels = load_data(csv_path, usecols, rename_cols, col_order=col_order)

# Plot data with EWM applied and updated labels
plot_perplexity_category(df, labels=updated_labels, span=400, name=name)



# # Bwd Fwd
# csv_path = "/home/zsarwar/Projects/Wandb/data/wandb_export_2025-01-31T02_19_57.571-06_00_fwd_bwd.csv"
# name = "fwd_bwd_ablation.pdf"

# # Define columns to load (Step, loss for default vector, loss for sparsemixer)
# usecols = [0, 1, 4]  
# rename_cols = ["Step", "Default MoE - Bwd", "Default MoE - Fwd + Bwd"]
# #col_order = ["Step", "TopK - 1024", "Default MoE - 1024"]
# # # Define new column names for cleaner plotting
# # label_map = {
# #     "Default MoE - 1024": "Default MoE - 1024",
# #     "TopK - 1024": "TopK - 1024",

# # }

# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols)

# # Plot data with EWM applied and updated labels
# plot_perplexity(df, labels=updated_labels, span=1000, name=name)



# # Bwd Fwd
# csv_path = "/home/zsarwar/Projects/gpt_neox/gpt-neox/code/default_analysis/Wandb/model_configs.csv"
# name = "comp_plots_ablation.pdf"

# Define columns to load (Step, loss for default vector, loss for sparsemixer)
#usecols = [0, 1, 4, 7, 10, 13, 16, 19, 22]
# usecols = [0, 1, 10, 4, 7, 19, 16, 13, 22]  

#usecols = [0, 4, 13, 7, 10, 25, 22, 19, 28]  
# usecols =  [0, 10, 13, 7, 4, 1, 25, 22, 19, 28, 16]

# rename_cols = ["Step", "Default MoE 8c1", "Default MoE 32c2", "Default MoE 32c1",  "Default MoE 8c2","Default MoE 32c4" "TopK 32c1", "TopK 8c2", "TopK 8c1", "TopK 32c2", "TopK 32c4"]
# col_order =   ["Step", "Default MoE 8c1", "Default MoE 8c2", "Default MoE 32c2", "Default MoE 32c1", "Default MoE 32c4", "TopK 8c1", "TopK 8c2", "TopK 32c1", "TopK 32c2", "TopK 32c4"]

# usecols = [0, 10, 13, 7, 4, 1, 25, 22, 19, 28, 16]  
# rename_cols = ["Step", "Default MoE 8c1", "Default MoE 32c2", "Default MoE 32c1", "Default MoE 8c2", "Default MoE 32c4","TopK 32c1", "TopK 8c2", "TopK 8c1", "TopK 32c2", "TopK 32c4"]
# col_order = ["Step", "Default MoE 8c1", "Default MoE 8c2", "Default MoE 32c2", "Default MoE 32c1", "Default MoE 32c4" "TopK 8c1",  "TopK 8c2", "TopK 32c1", "TopK 32c2", "TopK 32c4"]

# usecols = [0] + [1 + 3*i for i in range(10)]
# # → [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28]

# rename_cols = [
#     "Step",
#     "Default MoE 32c4",
#     "Default MoE 32c1",
#     "Default MoE 32c2",
#     "Default MoE 8c1",
#     "Default MoE 8c2",
#     "TopK 32c4",
#     "TopK 32c1",
#     "TopK 8c2",
#     "TopK 8c1",
#     "TopK 32c2",
# ]

# col_order = [
#     "Default MoE 32c4",
#     "Default MoE 32c1",
#     "Default MoE 32c2",
#     "Default MoE 8c1",
#     "Default MoE 8c2",
#     "TopK 32c4",
#     "TopK 32c1",
#     "TopK 8c2",
#     "TopK 8c1",
#     "TopK 32c2",
# ]


# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols, col_order=col_order)

# # Plot data with EWM applied and updated labels
# plot_perplexity_category(df, labels=updated_labels, span=300, name=name)






# # 2048 DIM
# csv_path = "/home/zsarwar/Projects/Wandb/data/wandb_export_2025-01-30T22_27_58.555-06_00_dim_2048.csv"
# name = "hidden_dim_2048_ablation.pdf"

# # Define columns to load (Step, loss for default vector, loss for sparsemixer)
# usecols = [0, 1, 4]  
# rename_cols = ["Step", "TopK - 2048", "Default MoE - 2048"]

# # Define new column names for cleaner plotting
# label_map = {
#     "TopK - 2048": "TopK - 2048",
#     "Default MoE - 2048": "Default MoE - 2048"
# }

# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols, label_map)

# # Plot data with EWM applied and updated labels
# plot_perplexity(df, labels=updated_labels, span=1000, name=name)


# # DIM 512
# csv_path = "/home/zsarwar/Projects/Wandb/data/wandb_export_2025-01-30T22_33_28.788-06_00_dim_512.csv"
# name = "hidden_dim_512_ablation.pdf"

# # Define columns to load (Step, loss for default vector, loss for sparsemixer)
# usecols = [0, 1, 4]  
# rename_cols = ["Step", "TopK - 512", "Default MoE - 512"]

# # Define new column names for cleaner plotting
# label_map = {
#     "TopK - 512": "TopK - 512",
#     "Default MoE - 512": "Default MoE - 512"
# }

# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols, label_map)
# # Plot data with EWM applied and updated labels
# plot_perplexity(df, labels=updated_labels, span=1000, name=name)



# # DIM 1024
# csv_path = "/home/zsarwar/Projects/Wandb/data/wandb_export_2025-01-30T22_27_58.555-06_00_dim_2048.csv"
# name = "hidden_dim_1024_ablation.pdf"

# # Define columns to load (Step, loss for default vector, loss for sparsemixer)
# usecols = [0, 1, 4]  
# rename_cols = ["Step",  "TopK","Default MoE",   ]
# # col_order = ["Step", "TopK", "Default MoE"]
# # # Define new column names for cleaner plotting
# # label_map = {
# #     "Default MoE - 1024": "Default MoE - 1024",
# #     "TopK - 1024": "TopK - 1024",

# # }

# # Load data with new column labels
# df, updated_labels = load_data(csv_path, usecols, rename_cols)


# # Define target perplexity
# target_perplexity = 36.0

# # Run the comparison
# result = compare_token_efficiency(df, updated_labels, target_perplexity=target_perplexity)

# # Print results
# print("Token Efficiency Comparison:")
# print(result)