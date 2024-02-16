from typing import Dict
import jax
import jax.numpy as jnp
import optax
from optax import OptState
from flax.training import train_state
from tqdm import tqdm
from models.additive import AdditiveAutoencoder
from models.parent import Autoencoder
from type_shorthands import *
from data import multimnist

import matplotlib.pyplot as plt

# Define the loss function
def mse_loss(params: Dict[str, float], model: Autoencoder, batch: R_bxdxdxc):
    # when declaring mutable, model.apply returns tuple, with first element
    # being actual model output
    reconstructions = model.apply(params, batch, mutable=['batch_stats'])[0]
    loss = jnp.mean((reconstructions - batch) ** 2)
    return loss

# Define a single training step
@jax.jit
def train_step(state: OptState, batch: R_bxdxdxc):
    loss_fn = lambda params: mse_loss(params, model, batch)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss

# Initialize the model and optimizer
ds = 10000 # Number of samples in dataset
latent_dim = 100  # Example latent dimension
num_blocks = 50   # Example number of blocks
key = jax.random.PRNGKey(0)

model = AdditiveAutoencoder(latent_dim, num_blocks)
params = model.init(key, jnp.ones((1, 100, 100, 3)))  # Dummy input for initialization

optimizer = optax.adam(learning_rate=1e-3)

# Initialize training state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

# Dataset
images = multimnist.generate(ds, export_jax=True)[0]

# Training loop
num_epochs = 20
batch_size = 100

for epoch in range(num_epochs):
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i+batch_size]
        state, loss = train_step(state, batch)

    print(f"Epoch {epoch}, Loss: {loss}")

# print reconstruction quality
# take a random image, and display (i) the image, (ii) the reconstruction, and (iii) the individual reconstructions for all blocks
image = images[0]
reconstruction = model.apply(state.params, image[None, ...], mutable=['batch_stats'])[0][0]
# renormalize to ints in [0, 255]
reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")


plt.subplot(1, 2, 2)
plt.imshow(reconstruction)
plt.title("Reconstruction")

# save current figure as image
plt.savefig("reconstruction.png")

# clear plot
plt.clf()

# for all 20 blocks, print individual decoding
z = model.apply(state.params, image[None, ...], mutable=['batch_stats'], method=model.encode)[0]
decoded_blocks = model.apply(state.params, z, mutable=['batch_stats'], method=model.decode_individual_blocks)[0]

fig = plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

for i in range(num_blocks):
    ax = fig.add_subplot(2, num_blocks // 2, i+1)
    # renormalize to [0, 1]
    decoded_blocks_show = (decoded_blocks[i] - decoded_blocks[i].min()) / (decoded_blocks[i].max() - decoded_blocks[i].min())
    ax.imshow(decoded_blocks_show[0])
    ax.set_title(f"Block {i}")

plt.tight_layout()  # Add this line to automatically adjust subplot positions

plt.savefig("blocks.png")
