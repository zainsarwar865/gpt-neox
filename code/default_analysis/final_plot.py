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


def plot_perplexity(df, labels, span=1000, name="plot.pdf"):

    plt.rcParams['figure.dpi'] = 300
    fontdict = {'family': 'serif', 'weight': 'normal', 'size': 10}
    plt.rc('font', **fontdict)

    plt.clf() 
    plt.figure(figsize=(4, 2.5))

    for label in labels:
        smoothed_col = label + "_EWM"
        df[smoothed_col] = df[label + "_Perplexity"].ewm(span=span).mean()
        plt.plot(df["Tokens"], df[smoothed_col], label=label, linestyle="solid")

    plt.xlabel("Tokens")
    plt.legend(loc="lower left")
    plt.grid(True)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e9:.0f}B'))


    perplexity_columns = [col for col in df.columns if col.endswith("_EWM")]


    y_min = df[perplexity_columns].min().min()
    plt.ylim(y_min * 0.97, 32)  # NEED 42 FOR 512


    plt.savefig(name, format="pdf", bbox_inches="tight")
    plt.show()







def extract_lr(label):
    match = re.search(r"\d+e-\d+", label)
    return match.group(0) if match else "Unknown"

def extract_category(label):
    return label.split(extract_lr(label))[0].strip() if "e-" in label else label

def plot_perplexity_category(df, labels, span=1000, name="plot.pdf"):
    plt.rcParams['figure.dpi'] = 300
    fontdict = {'family': 'serif', 'weight': 'normal', 'size': 10}
    plt.rc('font', **fontdict)
    category_styles = {}
    line_styles = ["solid", "dashed", "dotted", "dashdot"] 
    style_cycle = itertools.cycle(line_styles) 

    for label in labels:
        category = extract_category(label)
        if category not in category_styles:
            category_styles[category] = next(style_cycle)
    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    lr_colors = {}
    plt.figure(figsize=(4, 2.5))


    for label in labels:
        smoothed_col = label + "_EWM"
        df[smoothed_col] = df[label + "_Perplexity"].ewm(span=span).mean()

        category = extract_category(label)
        lr = extract_lr(label)

        if lr not in lr_colors:
            lr_colors[lr] = next(color_cycle)

        plt.plot(df["Tokens"], df[smoothed_col], label=label, linestyle=category_styles[category], color=lr_colors[lr], lw=0.75)

    plt.xlabel("Tokens")
    plt.grid(True)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e9:.0f}B'))

    perplexity_columns = [col for col in df.columns if col.endswith("_EWM")]

    y_min = df[perplexity_columns].min().min()
    plt.ylim(y_min * 0.97, 32) 

    category_legend = [plt.Line2D([0], [0], color="black", linestyle=category_styles[cat], lw=2, label=cat) for cat in category_styles]
    lr_legend = [plt.Line2D([0], [0], color=lr_colors[lr], linestyle="solid", lw=2, label=lr) for lr in lr_colors]
    plt.legend(handles=category_legend + lr_legend, loc="upper right", ncol=2)

    plt.savefig(name, format="pdf", bbox_inches="tight")
    plt.show()




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
# csv_path = "/home/zsarwar/Projects/Wandb/data/wandb_export_2025-01-30T22_49_02.952-06_00_dim_1024.csv"
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


# # LR Sweep

# csv_path = "/home/zsarwar/Projects/Wandb/data/wandb_export_2025-01-30T23_14_54.249-06_00_lr_sweep.csv"
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
# plot_perplexity_category(df, labels=updated_labels, span=1000, name=name)



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



# Bwd Fwd
csv_path = "/home/zsarwar/Projects/gpt_neox/gpt-neox/code/default_analysis/Wandb/model_part2.csv"
name = "comp_plots_ablation.pdf"

# Define columns to load (Step, loss for default vector, loss for sparsemixer)
#usecols = [0, 1, 4, 7, 10, 13, 16, 19, 22]
# usecols = [0, 1, 10, 4, 7, 19, 16, 13, 22]  
# rename_cols = ["Step", "Default MoE 8c1", "Default MoE 32c2", "Default MoE 32c1", "Default MoE 8c2", "TopK 32c1", "TopK 8c2", "TopK 8c1", "TopK 32c2"]
# col_order = ["Step", "Default MoE 8c1", "Default MoE 8c2", "Default MoE 32c2", "Default MoE 32c1",  "TopK 8c1",  "TopK 8c2", "TopK 32c1", "TopK 32c2"]

# pick out Step + the first column of each 3-column block
usecols = [0] + [1 + 3*i for i in range(10)]
# → [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28]

rename_cols = [
    "Step",
    "default vector 32c4 β=0.9999",
    "default vector 32c1 β=0.65",
    "default vector 32c2 β=0.95",
    "default vector 8c1 (zero-init EMA)",
    "default vector 8c2 (expertwise β=0.9)",
    "baseline 32c4",
    "baseline 32c1",
    "baseline 8c2",
    "baseline 8c1",
    "baseline 32c2",
]

col_order = [
    "default vector 32c4 β=0.9999",
    "default vector 32c1 β=0.65",
    "default vector 32c2 β=0.95",
    "default vector 8c1 (zero-init EMA)",
    "default vector 8c2 (expertwise β=0.9)",
    "baseline 32c4",
    "baseline 32c1",
    "baseline 8c2",
    "baseline 8c1",
    "baseline 32c2",
]


#col_order = ["Step", "TopK - 1024", "Default MoE - 1024"]
# # Define new column names for cleaner plotting
# label_map = {
#     "Default MoE - 1024": "Default MoE - 1024",
#     "TopK - 1024": "TopK - 1024",

# }

# Load data with new column labels
df, updated_labels = load_data(csv_path, usecols, rename_cols, col_order=col_order)

# Plot data with EWM applied and updated labels
plot_perplexity_category(df, labels=updated_labels, span=1000, name=name)






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