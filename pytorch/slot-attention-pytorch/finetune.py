import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CLEVR  # Assuming this is your dataset class
from model import SlotAttentionAutoEncoder  # Assuming this is your model class
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
import wandb
import os


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-trained model.")

    # Add necessary arguments here. Example:
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to save finetune checkpoints')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained model checkpoint')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to fine-tune for')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')

    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your model
    model = SlotAttentionAutoEncoder().to(device)
    model.load_state_dict(torch.load(opt.model_path)['model_state_dict'])

    # Freeze the model weights
    freeze_model(model)

    # Make modifications to your model here if necessary
    # For example, changing the final layer, unfreezing certain layers, etc.
    # model.new_layer = nn.Linear(...).to(device)
    # unfreeze_model(model.new_layer)  # Only train new_layer

    train_set = CLEVR('train')
    train_loader = DataLoader(
        train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    criterion = nn.MSELoss()

    # Setup WandB
    wandb.init(dir=os.path.abspath(opt.results_dir), project=f'{opt.model_name}_{opt.dataset_name}',
               job_type='train', mode='online')  # mode='offline'

    for epoch in range(opt.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{opt.num_epochs}], Loss: {avg_loss:.4f}")

        # Log to WandB
        wandb.log({'loss': total_loss}, step=epoch)

        # Save model checkpoint
        if epoch % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
            }, opt.results_dir + f"/{opt.model_name}.ckpt")


if __name__ == "__main__":
    main()
