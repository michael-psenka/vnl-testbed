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
    output_shape: tuple  # Expected output shape (H, W, C)

    def setup(self):
        self.encoder = EncoderCNN(num_features=self.slot_size)
        self.slot_attention = SlotAttentionModule(self.num_slots, self.slot_size, self.iters, self.mlp_hidden_size)
        self.decoder = DecoderCNN(self.output_shape)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def __call__(self, x):
        features = self.encoder(x)
        slots = self.slot_attention(features)
        reconstructed = self.decoder(slots)
        return reconstructed
    
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

        return x # [b, h*h, d]

class SlotAttentionModule(nn.Module):
    num_slots: int
    slot_size: int
    iters: int
    mlp_hidden_size: int

    @nn.compact
    def __call__(self, inputs):
        # Initialize slots
        slots = self.param('slots', nn.initializers.xavier_uniform(), (inputs.shape[0], self.num_slots, self.slot_size))

        # Slot attention iterations
        for _ in range(self.iters):
            prev_slots = slots
            # Compute attention logits
            attn_logits = jnp.einsum('bid, bjd -> bij', inputs, slots)
            attn = nn.softmax(attn_logits, axis=-1)

            # Weighted sum of features for each slot
            updates = jnp.einsum('bij, bid -> bjd', attn, inputs)

            # Update slots using a simple MLP
            slots, _ = nn.GRUCell(self.slot_size)(updates, prev_slots)
            updates = nn.Dense(self.mlp_hidden_size)(updates)
            updates = nn.relu(updates)
            updates = nn.Dense(self.slot_size)(updates)
            slots += updates

        # Convert to [batch*num_slots, slot_size]
        slots = jnp.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
        slots = jnp.tile(slots, [1, 8, 8, 1])

        return slots # [b, n, d]

class DecoderCNN(nn.Module):
    output_shape: tuple  # Expected output shape (H, W, C)

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

        # Upscale to the desired output shape
        # Adjust the number of layers, kernel size, and strides based on the specific output shape
        x = nn.ConvTranspose(features=self.output_shape[-1], kernel_size=(3,3), strides=(1,1), padding='SAME')(x)

        # Reshape and Recombine across all slots
        x = jnp.reshape(x, [self.output_shape[0], -1] + list(x.shape)[1:])
        x = jnp.sum(x, axis=1)

        # Final activation (e.g., sigmoid for images in [0, 1])
        x = nn.sigmoid(x)
        return x # [b, h, h, c]

__all__ = ["SlotAttentionAutoencoder"]