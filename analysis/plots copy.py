import numpy as np
import matplotlib.pyplot as plt

# Simulate tokens in billions
tokens_b = np.linspace(1.5, 6.0, 10)

# Simulate validation loss for different LORE ranks
# Higher rank = more LOREs, less capacity → higher loss
val_loss_rank1  = 4.3 - 0.229 * np.log(tokens_b)
val_loss_rank4  = 4.3 - 0.229 * np.log(tokens_b)
#val_loss_rank8  = 4.275 - 0.235 * np.log(tokens_b)
#val_loss_rank16 = 4.289 - 0.233 * np.log(tokens_b)

# Plot
plt.figure(figsize=(8, 6))

plt.plot(tokens_b, val_loss_rank1,  label="Additive Fusion", color="blue", linewidth=2)
plt.plot(tokens_b, val_loss_rank4,  label="GeGLU Fusion",  color="orange", linewidth=2)
# plt.plot(tokens_b, val_loss_rank8,  label="Rank 16 - 16 LOREs",  color="green", linewidth=2)
# plt.plot(tokens_b, val_loss_rank16, label="Rank 32 - 8 LOREs",  color="red", linewidth=2)

plt.title("Train Loss", fontsize=16)
plt.xlabel("Tokens (B)", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Save as high-quality PDF
plt.savefig("rank_lore_train_loss.pdf", dpi=300, bbox_inches="tight")
plt.show()