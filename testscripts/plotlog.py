import matplotlib.pyplot as plt
import numpy as np

# Load data from file
data = np.genfromtxt('check_points/hmaps1/log_edge.dat', delimiter=' ')

# Extract metrics
checkpoint = data[:, 0]
iteration = data[:, 1]
precision = data[:, 2]
recall = data[:, 3]
d_loss = data[:, 4]
g_loss = data[:, 5]
fm_loss = data[:, 6]
prec_non_zero = precision.nonzero()[0] # indices of non-zero precision values
rec_non_zero = recall.nonzero()[0] # indices of non-zero recall values

# Plot metrics
fig, axs = plt.subplots(4, figsize=(10, 10))
axs[0].plot(iteration[prec_non_zero], precision[prec_non_zero])
axs[0].set_title('Precision')
axs[1].plot(iteration[rec_non_zero], recall[rec_non_zero])
axs[1].set_title('Recall')
axs[2].plot(iteration, d_loss)
axs[2].set_title('Discriminator Loss')
axs[3].plot(iteration, g_loss, label='Generator Loss')
axs[3].plot(iteration, fm_loss, label='Feature Matching Loss')
axs[3].set_title('Generator Loss')
axs[3].legend()

# Save figure to file
fig.savefig('metrics.png')