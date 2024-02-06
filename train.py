import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from models.additive import AdditiveAutoencoder
from data import multimnist
from tqdm import tqdm

# Define the loss function
def mse_loss(params, model, batch):
    # when declaring mutable, model.apply returns tuple, with first element
    # being actual model output
    reconstructions = model.apply(params, batch, mutable=['batch_stats'])[0]
    loss = jnp.mean((reconstructions - batch) ** 2)
    return loss

# Define a single training step
@jax.jit
def train_step(state, batch):
    loss_fn = lambda params: mse_loss(params, model, batch)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss

# Initialize the model and optimizer
latent_dim = 100  # Example latent dimension
num_blocks = 50   # Example number of blocks

model = AdditiveAutoencoder(latent_dim, num_blocks)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 100, 100, 3)))  # Dummy input for initialization

optimizer = optax.adam(learning_rate=1e-3)

# Initialize training state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

# Dataset
images = multimnist.generate(10000, export_jax=True)[0]

# Training loop
num_epochs = 20
batch_size = 100

for epoch in range(num_epochs):
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i+batch_size]
        state, loss = train_step(state, batch)

    print(f"Epoch {epoch}, Loss: {loss}")
