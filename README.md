# Vision 'n Language Testbed

Welcome to the Vision and Language (VnL) Testbed repository. This is a testing ground and evaluation framework for various vision and language models, focused on controllable, synthetic datasets that allow for more rigorous and numerical evaluation on the learned vision/text alignment. As of current, the main model tested here is the [slot attention model](https://arxiv.org/abs/2006.15055).

## Getting Started 🚀

### Setup
1. Clone this repository:
```bash
git clone https://github.com/your-github-handle/vision-language-testbed
cd vision-language-testbed
```

2. Install necessary dependencies:
```bash
pip install -r requirements.txt
```

### Add New Vision+Language Models

* Ensure your model is saved under the `models` folder.
* Your model should inherit the `VisionLanguageModel` class located in [`models/parent.py`](models/parent/).
* Use [`models/clip.py`](models/clip.py) as a reference for implementing your model.

## Key Files 📂

- **[models/](models/)**: Contains the model implementations.
  - **[`clip.py`](models/clip.py)**: An implementation of a vision-language model using the CLIP architecture.
  - **[`slotattention.py`](models/slotattention.py)**: Implements the Slot Attention model, designed for unsupervised object-centric representation learning.
  - **[`parent/`](models/parent/)**: Contains base classes for models, particularly `autoencoder.py` and `vl.py` for vision-language tasks.

- **[data/](data/)**: Contains dataset generation scripts.
  - **[`multimnist.py`](data/multimnist.py)**: Generates the multi-MNIST dataset where each image comprises multiple MNIST digits with a descriptive text label.

- **[pytorch/slot-attention-pytorch/](pytorch/slot-attention-pytorch/)**: Contains training scripts and notebooks for the Slot Attention model.
  - **[`train.py`](pytorch/slot-attention-pytorch/train.py)**: A general training script integrating with the Slot Attention Autoencoder.
  - **[`train_tokencompressor.py`](pytorch/slot-attention-pytorch/train_tokencompressor.py)**: Training script for the token compressor model built off an iterated slot attention architecture, trained on a feature-to-feature autoencoding loss (akin to e.g. [CTRL](https://www.mdpi.com/1099-4300/24/4/456)).
  - **[`train_tokencompressor_direct.py`](pytorch/slot-attention-pytorch/train_tokencompressor_direct.py)**: Training script for the token compressor model built off an iterated slot attention architecture, trained on a data-to-data autoencoding loss.
  

## Datasets 📦

**multi-MNIST**:
The main controllable evaluation dataset. A constructor of "2D scenes", where each image comprises multiple MNIST digits with an associated text label. Labels describe spatial and visual attributes of the digits, such as "a 2 to the left of a red 4". The generator script allows for varying degrees of controllability.

## Example Script

Below is a simplified example script demonstrating how to use the Vision 'n Language Testbed repository to train a Slot Attention model on the multi-MNIST dataset. Please see the various training scripts linked above for more examples.

```python
import torch
from models.slotattention import SlotAttentionAutoEncoder
from data.multimnist import generate

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate dataset
num_samples = 1000
images, labels = generate(num_samples=num_samples)
images = torch.tensor(images, dtype=torch.float32).to(device)

# Model configuration
resolution = (128, 128)
num_slots = 7
num_iterations = 3
hidden_dim = 64
cnn_depth = 4
use_transformer = False

# Initialize the model
model = SlotAttentionAutoEncoder(
    resolution=resolution,
    num_slots=num_slots,
    num_iterations=num_iterations,
    hid_dim=hidden_dim,
    cnn_depth=cnn_depth,
    use_trfmr=use_transformer
).to(device)

# Training configuration
batch_size = 16
learning_rate = 0.0004
num_epochs = 10
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i in range(0, num_samples, batch_size):
        batch = images[i:i + batch_size]
        optimizer.zero_grad()
        recon_combined, recons, masks, slots = model(batch)
        loss = criterion(recon_combined, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / num_samples}")

# Save the model
torch.save(model.state_dict(), "slot_attention_model.pth")
```

This script sets up the dataset, configures the Slot Attention model, and runs a basic training loop. It is designed to help a newcomer understand how to utilize the repository's core functionalities.


## License 📄
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.