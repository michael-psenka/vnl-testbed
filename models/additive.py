from typing import List
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random
from .parent.autoencoder import Autoencoder

class AdditiveAutoencoder(Autoencoder):
    latent_dim: int
    num_blocks: int

    def setup(self):
        self.encoder = ResNet18Encoder(self.latent_dim)
        
        # Calculate block sizes
        block_size = self.latent_dim // self.num_blocks
        self.block_sizes = [block_size] * self.num_blocks
        
        # # Adjust the last block size to account for any remainder
        # self.block_sizes[-1] += self.latent_dim % self.num_blocks

        self.decoder_blocks = [DecoderBlock(block_size) for block_size in self.block_sizes]

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        decoded_blocks = self.decode_blocks(z)
        return jnp.sum(jnp.stack(decoded_blocks), axis=0)

    def decode_blocks(self, z):
        # construct list of points to split z at, based on block sizes

        z_blocks = jnp.split(z, self.num_blocks, axis=-1)
        decoded_blocks = [decoder_block(z_block) for decoder_block, z_block in zip(self.decoder_blocks, z_blocks)]
        return decoded_blocks

    @nn.compact
    def __call__(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

class DecoderBlock(nn.Module):
    block_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.leaky_relu(x, 0.01)

        for _ in range(5):
            x = nn.Dense(512)(x)
            x = nn.BatchNorm(use_running_average=False)(x)
            x = nn.leaky_relu(x, 0.01)

        # Define the Deconvolutional layers
        # Note: Flax does not have a direct 'DeConv' layer, so you might need to use ConvTranspose or similar.
        # Here's an example using ConvTranspose:

        batch_size = x.shape[0]
        x = x.reshape((batch_size, 8, 8, -1))  # Reshape to 4D tensor
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
        x = nn.leaky_relu(x, 0.01)
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
        x = nn.leaky_relu(x, 0.01)
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
        x = nn.leaky_relu(x, 0.01)
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
        x = nn.leaky_relu(x, 0.01)

        # Upscale to the desired output shape (channels)
        # Adjust the number of layers, kernel size, and strides based on the specific output shape
        x = nn.ConvTranspose(features=3, kernel_size=(3,3), strides=(1,1), padding='SAME')(x)

        return x



class ResNet18Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, kernel_size=(7, 7), strides=(1, 1), padding='SAME')(x)  # Reduced stride
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME')  # Reduced stride

        for features in [64, 128, 256, 512]:
            x = ResNetBlock(features)(x)
            x = ResNetBlock(features)(x)
            if features != 512:  # Downsampling
                x = ResNetBlock(features * 2, strides=(2, 2))(x)

        x = nn.avg_pool(x, window_shape=(4, 4))  # Adaptive pooling
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
    strides: tuple = (1, 1)

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
    
__all__ = ["AdditiveAutoencoder"]