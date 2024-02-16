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
        print("feature output")
        print(features.shape)
        slots = self.slot_attention(features)
        reconstructed = self.decoder(slots)
        return reconstructed

class SlotAttentionModule(nn.Module):
    num_slots: int
    slot_size: int
    iters: int
    mlp_hidden_size: int

    @nn.compact
    def __call__(self, inputs):
        # Initialize slots
        # num_slots = latent_dim*latent_dim, slot_size = downscaled feature size
        slots = self.param('slots', nn.initializers.xavier_uniform(), (inputs.shape[0], self.num_slots, self.slot_size))

        # Slot attention iterations
        for _ in range(self.iters):
            prev_slots = slots
            # Compute attention logits
            attn_logits = jnp.einsum('bid, bjd -> bij', inputs, slots)
            attn = nn.softmax(attn_logits, axis=-1)
            print(attn.shape)

            # Weighted sum of features for each slot
            updates = jnp.einsum('bij, bid -> bjd', attn, inputs)

            # Update slots using a simple MLP
            print(updates.shape)
            slots, _ = nn.GRUCell(self.slot_size)(updates, prev_slots)
            print(slots.shape)
            updates = nn.Dense(self.mlp_hidden_size)(updates)
            updates = nn.relu(updates)
            updates = nn.Dense(self.slot_size)(updates)
            slots += updates
            print(slots.shape)
            # print(updates.shape)
        
        # Reshape slots for compatibility with CNN decoder
        # E.g., if the decoder expects a 4D tensor (batch, height, width, channels),
        # you might need to reshape slots to this format.
        # N, D = inputs.shape[:2]
        # H, W = int(jnp.sqrt(D)), int(jnp.sqrt(D))  # Example: for square-shaped feature map
        # slots = slots.reshape((N, slots.shape[0], slots.shape[1]))
        # slots = slots[None, :, :]
        print('slots shape')
        print(slots.shape)

        # [batch*num_slots, slot_size]
        slots = jnp.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
        print(slots.shape)
        grid = jnp.tile(slots, [1, 8, 8, 1])
        print('grid')
        print(grid.shape)

        return slots
    
class EncoderCNN(nn.Module):
    num_features: int

    @nn.compact
    def __call__(self, x):
        # dimensions start with x.shape = (b, d, d, c)
        x = nn.Conv(features=self.num_features, kernel_size=(4,4), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_features, kernel_size=(4,4), padding='SAME')(x)
        x = nn.relu(x)
        
        # Flatten vector 
        x = jnp.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[-1]))

        # Final Linear Layer
        x = nn.Dense(self.num_features)(x)
        x = nn.BatchNorm(use_running_average=False)(x)

        return x

class DecoderCNN(nn.Module):
    output_shape: tuple  # Expected output shape (H, W, C)

    @nn.compact
    def __call__(self, x):
        # Assuming x is a flattened vector, reshape it to a suitable feature map
        # x = x.reshape((-1, 1, 1, self.output_shape[-1]))  # Reshape to (N, 1, 1, C)
        print('begin decoder ', x.shape)
        # Transposed convolution layers
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=(4, 4), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=(4, 4), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        # x = nn.ConvTranspose(features=64, kernel_size=(2,2), strides=(3, 3), padding='SAME')(x)
        # x = nn.relu(x)
        # x = nn.ConvTranspose(features=64, kernel_size=(2,2), strides=(2, 2), padding='SAME')(x)
        # x = nn.relu(x)

        # Upscale to the desired output shape
        # Adjust the number of layers, kernel size, and strides based on the specific output shape
        x = nn.ConvTranspose(features=self.output_shape[-1], kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
        print('end decoder ', x.shape)
        # Reshape
        x = jnp.reshape(x, [self.output_shape[0], -1] + list(x.shape)[1:])
        print('output ', x.shape)
        # Final activation (e.g., sigmoid for images in [0, 1])
        x = nn.sigmoid(x)
        print('output decoder')
        print(x.shape)

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

__all__ = ["SlotAttentionAutoencoder"]