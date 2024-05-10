import os
import math
import argparse
from dataset import *
from model import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
import wandb
from glob import glob

from utils.plotting import plot_samples

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='./tmp/model10.ckpt',
                        type=str, help='where to save models')
    parser.add_argument('--model_name', default='full-token-compressor',
                        type=str, help='model name')
    parser.add_argument('--loaded_model', default=None,
                        type=str, help='loaded model name')
    parser.add_argument('--dataset_name', default='CLEVR',
                        type=str, help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_slots', default=7, type=int,
                        help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int,
                        help='Number of attention iterations.')
    parser.add_argument('--hid_dim', default=64, type=int,
                        help='hidden dimension size')
    parser.add_argument('--learning_rate', default=0.0004, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int,
                        help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.5, type=float,
                        help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=100000, type=int,
                        help='Number of steps for the learning rate decay.')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for loading data')
    parser.add_argument('--num_epochs', default=1000, type=int,
                        help='number of workers for loading data')
    parser.add_argument('--cnn_depth', default=4, type=int,
                        help='number of encoder layers')
    parser.add_argument('--use_trfmr', default=False, type=bool,
                        help='use transformer in slot attention')
    parser.add_argument('--use_trfmr_encoder', default=False, type=bool,
                        help='use transformer encoder in slot attention')
    parser.add_argument('--use_trfmr_decoder', default=False, type=bool,
                        help='use transformer decoder in slot attention')
    parser.add_argument('--notes', type=str,
                        help='describe the change of this model')
    parser.add_argument('--ablated_indices', nargs='*', type=int,
                        help='number of layers to ablate')
    parser.add_argument('--max_samples', default=None, type=int,
                        help='max samples')
    parser.add_argument('--decay_type', default='cosine', type=str)
    parser.add_argument('--ckpt_epoch', default=10, type=int)
    parser.add_argument('--wandb', action='store_true',
                        help='whether to use wandb')
    parser.add_argument('--plot_recons', action='store_true',
                        help='whether to plot reconstructions each epoch')
    return parser.parse_args()


def create_model(opt):
    model_slotattention = SlotAttentionAutoEncoder(
        (128, 128), opt.num_slots, opt.num_iterations, opt.hid_dim, cnn_depth=opt.cnn_depth,
        use_trfmr=opt.use_trfmr, use_transformer_encoder=opt.use_trfmr_encoder, use_transformer_decoder=opt.use_trfmr_decoder).to(device)
    model = SlotAttentionCompressionAutoencoderDirect(
        model_slotattention, opt.num_slots, opt.hid_dim).to(device)

    if opt.loaded_model:
        model.load_state_dict(torch.load(
            f"{opt.results_dir}/{opt.loaded_model}.ckpt")['model_state_dict'])

    return model


def prepare_dataloader(batch_size, num_workers, dataset=None, max_samples=1000):
    dataset = dataset if dataset is not None else torch.ones(
        max_samples).to("cpu")
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader


def get_lr(step: int, total_steps: int, warmup_steps: int, lr: float, decay_type: str, decay_rate: float = None):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    else:
        if decay_type == 'cosine':
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr *= (1 + math.cos(math.pi * progress)) / 2
        elif decay_type == 'exp' and decay_rate is not None:
            lr *= decay_rate ** ((step - warmup_steps) / total_steps)
    return lr


def train(opt):
    model = create_model(opt)
    train_set = CLEVR('train')
    max_samples = opt.max_samples if opt.max_samples else len(train_set)
    image_dataloader = prepare_dataloader(
        opt.batch_size, opt.num_workers, train_set)
    z_dataloader = prepare_dataloader(
        opt.batch_size, opt.num_workers, max_samples=max_samples)
    criterion = nn.MSELoss()
    ablated_indices = set(opt.ablated_indices or [])
    # OUTER LOOP: number of slots we have currently compressed to
    for model_depth in range(opt.num_slots):
        if model_depth in ablated_indices:
            continue

        # Reset optimizer and scheduler for each model depth
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
        total_steps = opt.num_epochs * len(image_dataloader)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr(
                step, total_steps, opt.warmup_steps, opt.learning_rate, opt.decay_type, opt.decay_rate)
        )

        # Make results folder (holds all experiment subfolders)
        if opt.wandb:
            os.makedirs(opt.results_dir, exist_ok=True)
            opt.exp_index = len(glob(f"{opt.results_dir}/*"))
            model_filename = f"{opt.exp_index:03d}-{opt.model_name}-slots{opt.num_slots}-layer{opt.num_slots-model_depth}-{opt.notes}"
            wandb.init(dir=os.path.abspath(opt.results_dir), project=f"slot_att_pretrained", name=model_filename,
                       config=opt, job_type='train', mode='online', notes=opt.notes)
            torch.save({
                'model_state_dict': model.state_dict(),
            }, opt.results_dir + f"/{model_filename}.ckpt")

        start = time.time()
        # INNER LOOP: training the current step of the model
        for epoch in range(opt.num_epochs):
            model.train()
            total_loss = 0
            for x, z in tqdm(zip(image_dataloader, z_dataloader), total=len(z_dataloader)):
                x = x['image'].to(device)
                z = x if model_depth == 0 else z.to(device)
                # reconstruction of current feature
                x_hat = model.forward_step(z, model_depth)
                loss = criterion(x_hat, x)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            total_loss /= len(z_dataloader)

            if opt.wandb:
                wandb.log({'loss': total_loss}, step=epoch)
                if opt.plot_recons:
                    # log a plot of 10 sample images from the train set and their reconstructions
                    # Define the number of samples to visualize
                    first_batch_images = next(iter(image_dataloader))
                    first_batch_images = first_batch_images['image']
                    # Assuming that the batch size is at least as large as the number of samples you want to visualize
                    n = opt.batch_size if opt.batch_size < 10 else 10
                    sample_images = first_batch_images[:n].to(device)
                    # Note model.train() is already run before each loop
                    model.eval()
                    # Generate reconstructions at each depth
                    depths_reconstructions = [model.reconstruction_to(
                        sample_images, depth) for depth in range(model_depth+1)]
                    plt = plot_samples(sample_images, depths_reconstructions)
                    wandb.log({"Reconstructions by Depth": wandb.Image(
                        plt, caption=f"Epoch: {epoch + model_depth*opt.num_epochs}")}, step=epoch + model_depth*opt.num_epochs)
                    plt.close()
            if epoch % opt.ckpt_epoch:
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, opt.results_dir + f"/{model_filename}.ckpt")
              
            print("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
                                                         datetime.timedelta(seconds=time.time() - start)))

        # at the end of each training cycle, convert the training data to the compressed representation through
        # the currently trained encoder

        # we want to replace the dataloader with the compressed data
        z_new = []
        with torch.no_grad():
            model.eval()
            for x, z in zip(image_dataloader, z_dataloader):
                z = x['image'].to(device) if model_depth == 0 else z.to(device)
                z_fwd = model.get_compressed(
                    z, model_depth).detach().clone()
                z_new.append(z_fwd)
            z_new = torch.cat(z_new, dim=0).cpu()

        z_dataloader = prepare_dataloader(
            opt.batch_size, opt.num_workers, z_new)

        if opt.wandb:
            torch.save({
                'model_state_dict': model.state_dict(),
            }, opt.results_dir + f"/{model_filename}.ckpt")
            wandb.finish()
            


if __name__ == '__main__':
    opt = parse_args()
    train(opt)
