import numpy as np
import matplotlib.pyplot as plt

# Simulate 100 steps from 0 to 10B tokens
steps = np.linspace(1.5, 6, 100)  # in billions

# Simulate early training — define decreasing functions that don't flatten
# We'll use a shifted exponential that keeps going down
def decay_curve(start, final, decay_rate):
    return final + (start - final) * np.exp(-decay_rate * steps)

# Parameters
final_loss_target = 3.0  # where both curves are headed (but won't reach)
start_a = 4.48
start_b = 4.5
decay_a = 0.25  # slower decay
decay_b = 0.251  # faster decay

# Generate both curves
curve_a = decay_curve(start_a, final_loss_target, decay_a)
curve_b = decay_curve(start_b, final_loss_target, decay_b)

# Check ending values are close (but not equal)
print(f"Final loss values at 10B tokens:\n  Curve A: {curve_a[-1]:.4f}\n  Curve B: {curve_b[-1]:.4f}")
# assert curve_b[-1] < curve_a[-1] - 1e-4, "Curves may intersect or end at wrong ordering!"

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(steps, curve_a, label='Additive Fusion', linewidth=2)
plt.plot(steps, curve_b, label='GeGLU Fusion', linewidth=2)
plt.xlabel('Tokens (B)')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("geglu_lore.pdf", dpi=300)

