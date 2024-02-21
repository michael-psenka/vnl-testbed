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
    # output_shape: tuple  # Expected output shape (H, W, C)

    def setup(self):
        self.encoder = EncoderCNN(num_features=self.slot_size)
        self.slot_attention = SlotAttentionModule(self.num_slots, self.slot_size, self.iters, self.mlp_hidden_size)
        self.decoder = DecoderCNN()

    def encode(self, x):
        features = self.encoder(x)
        slots = self.slot_attention(features)
        return slots

    def decode(self, slots, batch_size):
        recons = self.decoder(slots)
        # Reshape and combine across all slots
        recons_combined, block_recons = reshape_and_combine(recons, batch_size)
        # Final activation (e.g., sigmoid for images in [0, 1])
        recons_combined = nn.sigmoid(recons_combined)
        return recons_combined, block_recons

    def __call__(self, x):
        features = self.encoder(x)
        # [b, h*h, d]
        print("features")
        print(features.shape)
        slots = self.slot_attention(features)
        # [b*n, h, w, d]
        print("slots")
        print(slots.shape)
        recons = self.decoder(slots)
        # [b*n, h, w, c]
        print("recons")
        print(recons.shape)

        # Reshape and combine across all slots
        recons_combined, _ = reshape_and_combine(recons, batch_size=x.shape[0])
        # Final activation (e.g., sigmoid for images in [0, 1])
        recons_combined = nn.sigmoid(recons_combined)
        return recons_combined
    
def reshape_and_combine(recons: jnp.ndarray, batch_size: int):
    recons = jnp.reshape(recons, [batch_size, -1] + list(recons.shape)[1:])
    recons_combined = jnp.sum(recons, axis=1)
    return recons_combined, recons

class EncoderCNN(nn.Module):
    num_features: int

    @nn.compact
    def __call__(self, x):
        # dimensions start with x.shape = (b, h, h, c)
        x = nn.Conv(features=self.num_features, kernel_size=(5,5), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_features, kernel_size=(5,5), padding='SAME')(x)
        x = nn.relu(x)
        
        # Flatten vector 
        x = jnp.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[-1]))

        # Final Linear Layer
        x = nn.Dense(self.num_features)(x)
        x = nn.BatchNorm(use_running_average=False)(x)

        return x

class SlotAttentionModule(nn.Module):
    num_slots: int
    slot_size: int
    iters: int
    mlp_hidden_size: int

    @nn.compact
    def __call__(self, inputs):
        # Initialize slots
        slots = self.param('slots', nn.initializers.xavier_uniform(), (self.num_slots, self.slot_size))

        # Slot attention iterations
        for _ in range(self.iters):
            prev_slots = slots
            # Compute attention logits
            attn_logits = jnp.einsum('bid, jd -> bij', inputs, slots)
            attn = nn.softmax(attn_logits, axis=-1)

            # Weighted sum of features for each slot
            updates = jnp.einsum('bij, bid -> jd', attn, inputs)

            # Update slots using a simple MLP
            slots, _ = nn.GRUCell(self.slot_size)(updates, prev_slots)
            updates = nn.Dense(self.mlp_hidden_size)(updates)
            updates = nn.relu(updates)
            updates = nn.Dense(self.slot_size)(updates)
            slots += updates

        # Convert to [batch*num_slots, slot_size]
        slots = jnp.broadcast_to(slots, (inputs.shape[0], self.num_slots, self.slot_size))
        slots = jnp.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
        slots = jnp.tile(slots, [1, 8, 8, 1]) # TODO: change '8' to argument

        return slots

class DecoderCNN(nn.Module):
    # output_shape: tuple  # Expected output shape (H, W, C)

    @nn.compact
    def __call__(self, x):
        # Transposed convolution layers
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
        x = nn.ConvTranspose(features=3, kernel_size=(3,3), strides=(1,1), padding='SAME')(x)

        return x 

__all__ = ["SlotAttentionAutoencoder"]