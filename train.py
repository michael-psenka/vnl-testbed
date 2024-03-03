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
    print(batch.shape)
    reconstructions = model.apply(params, batch, mutable=['batch_stats'])[0]
    # print(reconstructions)
    loss = jnp.mean((reconstructions - batch) ** 2)
    return loss, reconstructions

# Define a single training step


@jax.jit
def train_step(state: OptState, batch: R_bxdxdxc):
    def loss_fn(params): return mse_loss(params, model, batch)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, reconstructions), grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss, reconstructions


model_type = 'slot_attention'  # CHANGE HERE
# Initialize the model and optimizer
ds = 1000  # Number of samples in dataset
resolution = 128  # Example width and height dimension of image. DO NOT MODIFY
num_blocks = 1   # Example number of blocks/slots
batch_size = 5
key = jr.PRNGKey(0)
key, subkey = jr.split(key)
lr = 1e-3
# TODO: make the width and height variable
output_shape = (1, resolution, resolution, 3)
decoder_init_shape = (8, 8)

# Dataset
images = multimnist.generate(ds, export_jax=True)[0]
# images = images.astype(jnp.float32) / 255.0

# Load standard MNIST dataset
# from keras.datasets import mnist
# import numpy as np
# (images, _), _ = mnist.load_data()
# x_train = images.astype('float32') / 255.
# images = images[:, :, :, None]
# print(images[100:120])

# Load CLEVR dataset
# import tensorflow_datasets as tfds
# import tensorflow as tf
# ds = tfds.load("clevr:3.1.0", split="train", data_dir='~/../../shared/tensorflow_datasets')
# def filter_fn(example, max_n_objects=1):
#   """Filter examples based on number of objects.

#   The dataset only has feature values for visible/instantiated objects. We can
#   exploit this fact to count objects.

#   Args:
#     example: Dictionary of tensors, decoded from tf.Example message.
#     max_n_objects: Integer, maximum number of objects (excl. background) for
#       filtering the dataset.

#   Returns:
#     Predicate for filtering.
#   """
#   return tf.less_equal(tf.shape(example["objects"]["3d_coords"])[0],
#                         tf.constant(max_n_objects, dtype=tf.int32))

# ds = ds.filter(filter_fn)
# def preprocess_clevr(features, resolution, apply_crop=False,
#                      get_properties=True, max_n_objects=10):
#   """Preprocess CLEVR."""
#   image = tf.cast(features["image"], dtype=tf.float32)
#   image = ((image / 255.0) - 0.5) * 2.0  # Rescale to [-1, 1].

#   if apply_crop:
#     crop = ((29, 221), (64, 256))  # Get center crop.
#     image = image[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], :]

#   image = tf.image.resize(
#       image, resolution, method=tf.image.ResizeMethod.BILINEAR)
#   image = tf.clip_by_value(image, -1., 1.)

#   if get_properties:
#     # One-hot encoding of the discrete features.
#     size = tf.one_hot(features["objects"]["size"], 2)
#     material = tf.one_hot(features["objects"]["material"], 2)
#     shape_obj = tf.one_hot(features["objects"]["shape"], 3)
#     color = tf.one_hot(features["objects"]["color"], 8)
#     # Originally the x, y, z positions are in [-3, 3].
#     # We re-normalize them to [0, 1].
#     coords = (features["objects"]["3d_coords"] + 3.) / 6.
#     properties_dict = collections.OrderedDict({
#         "3d_coords": coords,
#         "size": size,
#         "material": material,
#         "shape": shape_obj,
#         "color": color
#     })

#     properties_tensor = tf.concat(list(properties_dict.values()), axis=1)

#     # Add a 1 indicating these are real objects.
#     properties_tensor = tf.concat(
#         [properties_tensor,
#          tf.ones([tf.shape(properties_tensor)[0], 1])], axis=1)

#     # Pad the remaining objects.
#     properties_pad = tf.pad(
#         properties_tensor,
#         [[0, max_n_objects - tf.shape(properties_tensor)[0],], [0, 0]],
#         "CONSTANT")

#     features = {
#         "image": image,
#         "target": properties_pad
#     }

#   else:
#     features = {"image": image}

#   return features
# def _preprocess_fn(x, resolution, max_n_objects=max_n_objects):
#   return preprocess_clevr(
#       x, resolution, apply_crop=apply_crop, get_properties=get_properties,
#       max_n_objects=max_n_objects)
# ds = ds.map(lambda x: _preprocess_fn(x, resolution))
# ds = ds.batch(batch_size, drop_remainder=True)
# it = iter(tfds.as_numpy(ds))  # Convert the TensorFlow dataset to an iterator of NumPy arrays
# batch = next(it)  # Fetch one batch
# images = batch['image']
# images = images.astype('float32') / 255.
# print("CLEVER")
# print(images.shape)

if model_type == 'slot_attention':
    model = SlotAttentionAutoencoder(num_slots=num_blocks, slot_size=64, iters=3,
                                     mlp_hidden_size=128, decoder_init_shape=decoder_init_shape, output_shape=output_shape)
else:
    model = AdditiveAutoencoder(resolution, num_blocks)

params = model.init(key, jr.normal(subkey, output_shape)
                    )  # Dummy input for initialization

optimizer = optax.adam(learning_rate=lr)

# Initialize training state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i+batch_size]
        state, loss, reconstructions = train_step(state, batch)
        print(reconstructions)

    print(f"Epoch {epoch}, Loss: {loss}")

print("inference")
# print reconstruction quality
# take a random image, and display (i) the image, (ii) the reconstruction, and (iii) the individual reconstructions for all blocks
image = images[0]
reconstruction = model.apply(
    state.params, image[None, ...], mutable=['batch_stats'])[0][0]
# renormalize to ints in [0, 255]
# clip_value = jnp.percentile(reconstruction, 99)  # For example, clipping to the 99th percentile
# reconstruction = jnp.clip(reconstruction, None, clip_value)
# print("After clipping - Min:", reconstruction.min(), "Max:", reconstruction.max())
reconstruction = (reconstruction - reconstruction.min()) / \
    (reconstruction.max() - reconstruction.min())
print(reconstruction)
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.gray()
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(reconstruction)
plt.gray()
plt.title("Reconstruction")

# save current figure as image
plt.savefig("reconstruction.png")

# clear plot
plt.clf()

# for all 20 blocks, print individual decoding
z = model.apply(state.params, image[None, ...], mutable=[
                'batch_stats'], method=model.encode)[0]
decoded_blocks = model.apply(state.params, z, mutable=[
                             'batch_stats'], method=model.decode_blocks)[0]
decoded_blocks = jnp.squeeze(decoded_blocks, axis=0)  # Remove batch dimension
print(decoded_blocks.shape)
for i in range(num_blocks):
    ax = plt.subplot(2, 4, i+1)
    # renormalize to [0, 1]
    decoded_blocks_show = (decoded_blocks[i] - decoded_blocks[i].min()) / (
        decoded_blocks[i].max() - decoded_blocks[i].min())
    ax.imshow(decoded_blocks_show)
    ax.set_title(f"Block {i}")

plt.tight_layout()  # Add this line to automatically adjust subplot positions

plt.savefig("blocks.png")
