import timm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CLEVR  # Assuming this is your dataset class
from model import SlotAttentionAutoEncoder  # Assuming this is your model class
from torchvision.transforms import Compose, ToTensor, Resize
import wandb
import os
from glob import glob
from tqdm import tqdm
import time
import datetime


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


def stack_model(image, model, batch_size=32):
    z = model.encode(image)
    slots_k = model.slot_attention(z)
    slots_kminus1 = slots_k[:, :-1, :]
    # Instead of focusing attention spatially across the image, the model now allocates attention among conceptual entities represented by slots.
    slots_kminus1 = model.slot_attention(slots_kminus1)
    return model.decode(slots_kminus1, batch_size)


parser = argparse.ArgumentParser(
    description="Fine-tune a pre-trained model.")

# Add necessary arguments here. Example:
parser.add_argument('--results_dir', type=str, required=True,
                    help='Path to save finetune checkpoints')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to the pre-trained model checkpoint')
parser.add_argument('--finetuned_model_name', type=str, required=True,
                    help='Name of the finetuned model')
parser.add_argument('--dataset_name', default='CLEVR',
                    type=str, help='dataset')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for training')
parser.add_argument('--num_slots', default=7, type=int,
                    help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int,
                    help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int,
                    help='hidden dimension size')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of epochs to fine-tune for')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='Learning rate for fine-tuning')
parser.add_argument('--warmup_steps', default=10000, type=int,
                    help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float,
                    help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int,
                    help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers for data loading')
parser.add_argument('--cnn_depth', default=4, type=int,
                    help='number of encoder layers')
parser.add_argument('--use_trfmr', default=False, type=bool,
                    help='use transformer in slot attention')
parser.add_argument('--use_trfmr_encoder', default=False, type=bool,
                    help='use transformer encoder in slot attention')
parser.add_argument('--use_trfmr_decoder', default=False, type=bool,
                    help='use transformer decoder in slot attention')


opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your model
resolution = (128, 128)
frozen_model = SlotAttentionAutoEncoder(
    resolution, opt.num_slots, opt.num_iterations, opt.hid_dim, cnn_depth=opt.cnn_depth,
    use_trfmr=opt.use_trfmr, use_transformer_encoder=opt.use_trfmr_encoder, use_transformer_decoder=opt.use_trfmr_decoder).to(device)
frozen_model.load_state_dict(torch.load(opt.model_path)['model_state_dict'])

# Freeze the model weights
frozen_model.eval()

# Initialize a new model
model = SlotAttentionAutoEncoder(
    resolution, opt.num_slots, opt.num_iterations, opt.hid_dim).to(device)
# Set the frozen weights to the new model
model.fc1 = frozen_model.fc1
model.fc2 = frozen_model.fc2
model.encoder = frozen_model.encoder
model.decoder = frozen_model.decoder
model.slot_attention = frozen_model.slot_attention

# Freeze the specific components
for name, param in model.named_parameters():
    if "fc1" in name or "fc2" in name or "encoder" in name or "decoder" in name or "slot_attention" in name:
        param.requires_grad = False

train_set = CLEVR('train')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                                           shuffle=True, num_workers=opt.num_workers)

# params = [{'params': model.parameters()}]

optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                       model.parameters()), lr=opt.learning_rate)
criterion = nn.MSELoss()

# Make results folder (holds all experiment subfolders)
os.makedirs(opt.results_dir, exist_ok=True)
experiment_index = len(glob(f"{opt.results_dir}/*"))
# model_string_name = opt.model.replace("/", "-")
# Create an experiment folder
model_filename = f"{experiment_index:03d}-{opt.finetuned_model_name}"
# Setup WandB
wandb.init(dir=os.path.abspath(opt.results_dir), project=f"{opt.finetuned_model_name}_{opt.dataset_name}", name=model_filename,
           config=opt, job_type='train', mode='online')  # mode='offline'

start = time.time()
i = 0
for epoch in range(opt.num_epochs):
    model.train()

    total_loss = 0

    for sample in tqdm(train_loader):
        i += 1

        if i < opt.warmup_steps:
            learning_rate = opt.learning_rate * (i / opt.warmup_steps)
        else:
            learning_rate = opt.learning_rate

        learning_rate = learning_rate * (opt.decay_rate ** (
            i / opt.decay_steps))

        optimizer.param_groups[0]['lr'] = learning_rate

        image = sample['image'].to(device)
        recon_combined, recons, masks, slots = stack_model(
            image, model, batch_size=opt.batch_size)
        loss = criterion(recon_combined, image)
        total_loss += loss.item()

        del recons, masks, slots

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss /= len(train_loader)

    wandb.log({'loss': total_loss}, step=epoch)

    print("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
                                                 datetime.timedelta(seconds=time.time() - start)))

    if not epoch % 10:
        torch.save({
            'model_state_dict': model.state_dict(),
        }, opt.results_dir + f"/{model_filename}.ckpt")
