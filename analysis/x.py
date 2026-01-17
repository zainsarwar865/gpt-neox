import matplotlib.pyplot as plt
import numpy as np

# 1. Setup the x-axis (Tokens in Billions)
x = np.linspace(1.5, 6, 100)

# 2. Define a function to generate "loss-like" curves
# We use: Loss = base_offset + amplitude * exp(-decay * x)
def loss_curve(x, offset, amp, decay):
    return offset + amp * np.exp(-decay * x)

# 3. Create synthetic data for each rank (approximating your plot's values)
# Rank 1 - 256 LOREs (Blue) - Highest loss
y_rank1 = loss_curve(x, 3.80, 0.70, 0.3) 
# Rank 8 - 32 LOREs (Orange)
y_rank8 = loss_curve(x, 3.75, 0.75, 0.35)
# Rank 32 - 8 LOREs (Red)
y_rank32 = loss_curve(x, 3.73, 0.78, 0.37)
# Rank 16 - 16 LOREs (Green) - Lowest loss based on your image
y_rank16 = loss_curve(x, 3.71, 0.82, 0.4)

# 4. Plotting
plt.figure(figsize=(10, 7.5), dpi=100, edgecolor='black', linewidth=2)

plt.plot(x, y_rank1,  color='blue',   linewidth=2, label='Rank 1 - 256 DSMs')
plt.plot(x, y_rank8,  color='orange', linewidth=2, label='Rank 8 - 32 DSMs')
plt.plot(x, y_rank16, color='#38761d', linewidth=2, label='Rank 16 - 16 DSMs') # Forest green
plt.plot(x, y_rank32, color='#e03e2d', linewidth=2, label='Rank 32 - 8 DSMs') # Reddish

# 5. Styling to match your image
plt.title('Train Loss', fontsize=18)
plt.xlabel('Tokens (B)', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12, loc='upper right')

# Adjust ticks to match the image intervals
plt.xticks([2, 3, 4, 5, 6])
plt.yticks(np.arange(3.85, 4.30, 0.05))
plt.savefig("rank_lore_train_loss.pdf", bbox_inches='tight', dpi=300)
plt.show()