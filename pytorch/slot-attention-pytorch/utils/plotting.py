from matplotlib import pyplot as plt
import torchvision.utils as vutils
import numpy as np

def plot_samples(original, depths_reconstructions, num_samples=10):
    # Combine original and all reconstructions into one list
    images_to_plot = [original[:num_samples]] + [rec[:num_samples] for rec in depths_reconstructions]
    # Number of rows is original + number of depths
    num_rows = len(images_to_plot)
    fig, axes = plt.subplots(nrows=num_rows, figsize=(15, 2 * num_rows), gridspec_kw={'wspace':0.1, 'hspace':0.1})
    for i, images in enumerate(images_to_plot):
        # Create a grid of images for each row
        grid = vutils.make_grid(images, nrow=num_samples, normalize=True, scale_each=True)
        npgrid = grid.cpu().numpy()
        ax = axes[i] if num_rows > 1 else axes
        ax.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
        ax.axis('off')
        if i == 0:
            ax.set_title("Original")
        else:
            ax.set_title(f"Depth={i-1}")

    # plt.tight_layout()
    return plt