import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random
from typing import List
from models.parent.autoencoder import Autoencoder

class SlotAttentionAutoencoder(Autoencoder):
    num_slots: int
    slot_size: int
    iters: int
    mlp_hidden_size: int
    output_shape: tuple  # Expected output shape (H, W, C)

    def setup(self):
        self.encoder = EncoderCNN()
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

class SlotAttentionModule(nn.Module):
    num_slots: int
    slot_size: int
    iters: int
    mlp_hidden_size: int

    @nn.compact
    def __call__(self, features):
        # Initialize slots
        slots = self.param('slots', nn.initializers.xavier_uniform(), (self.num_slots, self.slot_size))

        # Slot attention iterations
        for _ in range(self.iters):
            # Compute attention logits
            attn_logits = jnp.einsum('bid, jd -> bij', features, slots)
            attn = nn.softmax(attn_logits, axis=-1)

            # Weighted sum of features for each slot
            updates = jnp.einsum('bij, bid -> jd', attn, features)

            # Update slots using a simple MLP
            slots = slots + nn.Dense(self.mlp_hidden_size)(updates)
            slots = nn.relu(slots)
            slots = nn.Dense(self.slot_size)(slots)

        # Reshape slots for compatibility with CNN decoder
        # E.g., if the decoder expects a 4D tensor (batch, height, width, channels),
        # you might need to reshape slots to this format.
        N, D = features.shape[:2]
        H, W = int(jnp.sqrt(D)), int(jnp.sqrt(D))  # Example: for square-shaped feature map
        slots = slots.reshape((N, H, W, -1))

        return slots

class DecoderCNN(nn.Module):
    output_shape: tuple  # Expected output shape (H, W, C)

    @nn.compact
    def __call__(self, x):
        # Assuming x is a flattened vector, reshape it to a suitable feature map
        x = x.reshape((-1, 1, 1, self.output_shape[-1]))  # Reshape to (N, 1, 1, C)

        # Transposed convolution layers
        x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)

        # Upscale to the desired output shape
        # Adjust the number of layers, kernel size, and strides based on the specific output shape
        x = nn.ConvTranspose(features=self.output_shape[-1], kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)

        # Final activation (e.g., sigmoid for images in [0, 1])
        x = nn.sigmoid(x)

        return x

class ResNet18Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        for features in [64, 128, 256, 512]:
            x = ResNetBlock(features)(x)
            x = ResNetBlock(features)(x)
            if features != 512:  # Downsampling
                x = ResNetBlock(features * 2, strides=(2, 2))(x)

        x = nn.avg_pool(x, window_shape=(7, 7))
        x = x.reshape((x.shape[0], -1))  # Flatten

        # Fully connected layers
        for _ in range(5):
            x = nn.Dense(512)(x)
            x = nn.BatchNorm(use_running_average=False)(x)
            x = nn.leaky_relu(x, 0.01)

        # Final Linear Layer
        x = nn.Dense(self.latent_dim)(x)
        x = nn.BatchNorm(use_running_average=False)(x)

        return x

class ResNetBlock(nn.Module):
    features: int
    strides = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(self.features, kernel_size=(3, 3), strides=self.strides)(x)
        y = nn.BatchNorm(use_running_average=False)(y)
        y = nn.relu(y)
        y = nn.Conv(self.features, kernel_size=(3, 3))(y)
        y = nn.BatchNorm(use_running_average=False)(y)

        if residual.shape != y.shape:
            residual = nn.Conv(self.features, kernel_size=(1, 1), strides=self.strides)(residual)
            residual = nn.BatchNorm(use_running_average=False)(residual)

        return nn.relu(y + residual)


