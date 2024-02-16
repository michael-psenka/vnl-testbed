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

# Define the loss function
def mse_loss(params: Dict[str, float], model: Autoencoder, batch: R_bxdxdxc):
    # when declaring mutable, model.apply returns tuple, with first element
    # being actual model output
    print('batch input')
    print(batch.shape)
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
latent_dim = 100  # Example latent dimension
num_blocks = 7   # Example number of blocks/slots
batch_size = 100
key = jr.PRNGKey(0)
key, subkey = jr.split(key)

if model_type == 'slot_attention':
    output_shape = (batch_size, 100, 100, 3) # TODO: make the width and height variable
    model = SlotAttentionAutoencoder(num_slots=num_blocks, slot_size=64, iters=1, mlp_hidden_size=128, output_shape=output_shape)
else:
    output_shape = (batch_size, latent_dim, latent_dim, 3)
    model = AdditiveAutoencoder(latent_dim, num_blocks)

params = model.init(key, jr.normal(subkey, output_shape))  # Dummy input for initialization

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

for epoch in range(num_epochs):
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i+batch_size]
        state, loss = train_step(state, batch)

    print(f"Epoch {epoch}, Loss: {loss}")
