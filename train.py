from typing import Dict
import jax
from jax import random as jr
import jax.numpy as jnp
import optax
from optax import OptState
from flax.training import train_state
from tqdm import tqdm
from models import *
from models.parent.autoencoder import Autoencoder
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

model_type = 'slot_attention' # CHANGE HERE
# Initialize the model and optimizer
ds = 1000 # Number of samples in dataset
resolution = 128  # Example width and height dimension of image. DO NOT MODIFY
num_blocks = 7   # Example number of blocks/slots
batch_size = 50
key = jr.PRNGKey(0)
key, subkey = jr.split(key)
lr = 1e-3
output_shape = (1, resolution, resolution, 3) # TODO: make the width and height variable

if model_type == 'slot_attention':
    model = SlotAttentionAutoencoder(num_slots=num_blocks, slot_size=64, iters=1, mlp_hidden_size=128)
else:
    model = AdditiveAutoencoder(resolution, num_blocks)

params = model.init(key, jr.normal(subkey, output_shape))  # Dummy input for initialization

optimizer = optax.adam(learning_rate=lr)

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

for epoch in range(num_epochs):
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i+batch_size]
        state, loss = train_step(state, batch)

    print(f"Epoch {epoch}, Loss: {loss}")

print("inference")
# print reconstruction quality
# take a random image, and display (i) the image, (ii) the reconstruction, and (iii) the individual reconstructions for all blocks
image = images[0]
reconstruction = model.apply(state.params, image[None, ...], mutable=['batch_stats'])[0][0]
# renormalize to ints in [0, 255]
clip_value = jnp.percentile(reconstruction, 99)  # For example, clipping to the 99th percentile
reconstruction = jnp.clip(reconstruction, None, clip_value)
reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
print(reconstruction)
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
decoded_im, decoded_blocks = model.apply(state.params, z, 1, mutable=['batch_stats'], method=model.decode)[0]
print(decoded_blocks.shape)
# fig = plt.figure(figsize=(20, 12))  # Adjust the figure size as needed

decoded_blocks = jnp.squeeze(decoded_blocks, axis=0)
print(decoded_blocks.shape)
for i in range(num_blocks):
    ax = plt.subplot(2, 4, i+1)
    # renormalize to [0, 1]
    decoded_blocks_show = (decoded_blocks[i] - decoded_blocks[i].min()) / (decoded_blocks[i].max() - decoded_blocks[i].min())
    ax.imshow(decoded_blocks_show)
    ax.set_title(f"Block {i}")

plt.tight_layout()  # Add this line to automatically adjust subplot positions

plt.savefig("blocks.png")
