import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random
from typing import List
from .parent.autoencoder import Autoencoder

class SlotAttentionAutoencoder(Autoencoder):
    num_slots: int
    slot_size: int
    iters: int
    mlp_hidden_size: int
    decoder_init_shape: tuple
    output_shape: tuple

    def setup(self):
        self.encoder = EncoderCNN(num_features=self.slot_size)
        self.slot_attention = SlotAttentionModule(self.num_slots, self.slot_size, self.iters, self.mlp_hidden_size, self.decoder_init_shape)
        self.decoder = DecoderCNN(num_channels=self.output_shape[-1])

        self.encoder_pos = SoftPositionEmbed(self.slot_size, (self.output_shape[1],self.output_shape[1]))
        self.decoder_pos = SoftPositionEmbed(self.slot_size, self.decoder_init_shape)
        
        self.mlp = nn.Sequential([nn.Dense(self.slot_size), nn.relu, nn.Dense(self.slot_size)])
        self.layer_norm = nn.LayerNorm()

    def encode(self, x):
        x = self.encoder(x)
        # [b, h, h, d]
        # [b, h*h, d]
        x = self.encoder_pos(x)
        
        # Flatten vector 
        x = jnp.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[-1]))
        
        x = self.mlp(self.layer_norm(x))
        # Final Linear Layer
        # x = nn.Dense(self.slot_size)(x)
        # x = nn.BatchNorm(use_running_average=False)(x)
        
        return x

    def decode(self, slots, batch_size=1):
        recons = self.decoder(slots)
        # [b*n, h, w, c]
        # Reshape and combine across all slots
        block_recons = reshape_recons(recons, batch_size, self.output_shape[-1])
        # masks = nn.softmax(masks, axis=1)
        # recon_combined = tf.reduce_sum(block_recons * masks, axis=1)  # Recombine image.
        recons_combined = jnp.sum(block_recons, axis=1)
        # Final activation (e.g., sigmoid for images in [0, 1])
        # recons_combined = nn.sigmoid(recons_combined)
        return recons_combined
    
    def decode_blocks(self, slots, batch_size=1):
        recons = self.decoder(slots)
        # Reshape and combine across all slots
        block_recons = reshape_recons(recons, batch_size, self.output_shape[-1])
        return block_recons

    def __call__(self, x):
        features = self.encode(x)
        slots = self.slot_attention(features)
        # [b*n, h, w, d]
        slots = self.decoder_pos(slots)
        recons_combined = self.decode(slots, batch_size=x.shape[0])
        return recons_combined
    
def reshape_recons(recons: jnp.ndarray, batch_size: int, num_channels=3):
    block_recons = jnp.reshape(recons, [batch_size, -1] + list(recons.shape)[1:])
    print(block_recons.shape)
    # REMOVED MASKS
    # block_recons, masks = jnp.split(block_recons, [num_channels, 1], axis=-1)
    return block_recons

class EncoderCNN(nn.Module):
    num_features: int

    @nn.compact
    def __call__(self, x):
        # dimensions start with x.shape = (b, h, h, c)
        print(x.shape)
        x = nn.Conv(features=self.num_features, kernel_size=(5,5), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_features, kernel_size=(5,5), padding='SAME')(x)
        x = nn.relu(x)

        return x

class SlotAttentionModule(nn.Module):
    num_slots: int
    slot_size: int
    iters: int
    mlp_hidden_size: int
    decoder_init_shape: tuple

    @nn.compact
    def __call__(self, inputs):
        # Normalize inputs
        inputs = nn.LayerNorm()(inputs)
        rng = jax.random.PRNGKey(0)
        
        # Initialize slots
        # slots = self.param('slots', nn.initializers.xavier_uniform(), (self.num_slots, self.slot_size))
        # Slot initialization parameters (similar to slots_mu + exp(slots_log_sigma))
        slots_mu = self.param('slots_mu', nn.initializers.xavier_uniform(), (self.num_slots, self.slot_size))
        slots_log_sigma = self.param('slots_log_sigma', nn.initializers.xavier_uniform(), (self.num_slots, self.slot_size))
        # Initialize slots
        slots = slots_mu + jnp.exp(slots_log_sigma) * random.normal(rng, (self.num_slots, self.slot_size))

        # Slot attention iterations
        for _ in range(self.iters):
            prev_slots = slots
            # normalize slots
            slots = nn.LayerNorm()(slots)
            # Compute attention logits
            attn_logits = jnp.einsum('bid, jd -> bij', inputs, slots)
            attn = nn.softmax(attn_logits, axis=-1)

            # Weighted sum of features for each slot
            updates = jnp.einsum('bij, bid -> jd', attn, inputs)

            # Update slots using a simple MLP
            slots, _ = nn.GRUCell(self.slot_size)(updates, prev_slots)
            updates = nn.LayerNorm()(updates)
            updates = nn.Dense(self.mlp_hidden_size)(updates)
            updates = nn.relu(updates)
            updates = nn.Dense(self.slot_size)(updates)
            slots += updates

        # Convert to [batch*num_slots, slot_size]
        slots = jnp.broadcast_to(slots, (inputs.shape[0], self.num_slots, self.slot_size))
        slots = jnp.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
        slots = jnp.tile(slots, [1, self.decoder_init_shape[0], self.decoder_init_shape[0], 1])

        return slots

class DecoderCNN(nn.Module):
    num_channels: int

    @nn.compact
    def __call__(self, x):
        # Transposed convolution layers
        # Used for MNIST dataset
        # x = nn.ConvTranspose(features=64, kernel_size=(3,3), strides=(2,2), padding='SAME')(x)
        # x = nn.relu(x)
        # x = nn.ConvTranspose(features=64, kernel_size=(3,3), strides=(2,2), padding='SAME')(x)
        # x = nn.relu(x)
        
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
        x = nn.relu(x)

        # Upscale to the desired output shape (channels)
        # Adjust the number of layers, kernel size, and strides based on the specific output shape
        x = nn.ConvTranspose(features=self.num_channels, kernel_size=(3,3), strides=(1,1), padding='SAME')(x)

        return x 

class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""
    hidden_size: int
    resolution: tuple

    @nn.compact
    def __call__(self, inputs):
        # Initialize the position grid
        grid = build_grid(self.resolution)
        grid = jnp.array(grid)  # Convert the numpy grid to a JAX array

        # Define a dense layer for projection
        projected_grid = nn.Dense(features=self.hidden_size)(grid)

        # Add the projected position grid to the inputs
        return inputs + projected_grid

def build_grid(resolution):
    """Builds a grid for soft positional embedding."""
    ranges = [jnp.linspace(0., 1., num=res) for res in resolution]
    grid = jnp.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = jnp.stack(grid, axis=-1)
    grid = jnp.reshape(grid, [resolution[0], resolution[1], -1])
    grid = jnp.expand_dims(grid, axis=0)
    grid = grid.astype(jnp.float32)
    return jnp.concatenate([grid, 1.0 - grid], axis=-1)


__all__ = ["SlotAttentionAutoencoder"]