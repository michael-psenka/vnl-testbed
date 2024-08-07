import os
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
parser.add_argument('--max_samples', default=1000, type=int,
                    help='max samples')
parser.add_argument('--wandb', action='store_true',
                    help='whether to use wandb')


opt = parser.parse_args()
resolution = (128, 128)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_slotattention = SlotAttentionAutoEncoder(
    resolution, opt.num_slots, opt.num_iterations, opt.hid_dim, cnn_depth=opt.cnn_depth,
    use_trfmr=opt.use_trfmr, use_transformer_encoder=opt.use_trfmr_encoder, use_transformer_decoder=opt.use_trfmr_decoder).to(device)

model = SlotAttentionCompressionAutoencoder(
    model_slotattention, opt.num_slots, opt.hid_dim).to(device)

if opt.loaded_model:
    model.load_state_dict(torch.load(
        f"/shared/rzhang/slot_att/tmp/{opt.loaded_model}.ckpt")['model_state_dict'])

train_set = CLEVR('train')
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                                               shuffle=True, num_workers=opt.num_workers)

criterion = nn.MSELoss()
params = [{'params': model.parameters()}]
optimizer = optim.Adam(params, lr=opt.learning_rate)

ablated_indices = set(opt.ablated_indices) if opt.ablated_indices else set()
max_samples = opt.max_samples if len(
    train_dataloader) > opt.max_samples else len(train_dataloader)
# OUTER LOOP: number of slots we have currently compressed to

for model_depth in range(opt.num_slots):
    if model_depth in ablated_indices:
        continue

    # Make results folder (holds all experiment subfolders)
    os.makedirs(opt.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{opt.results_dir}/*"))
    model_filename = f"{experiment_index:03d}-{opt.model_name}-slots{opt.num_slots}-{opt.notes}"

    torch.save({
        'model_state_dict': model.state_dict(),
    }, opt.results_dir + f"/{model_filename}.ckpt")
    opt.index = experiment_index
    if opt.wandb:
        wandb.init(dir=os.path.abspath(opt.results_dir), project=f"slot_att_pretrained", name=model_filename,
                   config=opt, job_type='train', mode='online', notes=opt.notes)

    start = time.time()
    i = 0

    # INNER LOOP: training the current step of the model
    for epoch in range(opt.num_epochs):
        model.train()

        total_loss = 0
        idx = 0
        for z in tqdm(train_dataloader):
            print(type(z))
            if idx >= max_samples:
                break
            idx += 1
            i += 1

            if i < opt.warmup_steps:
                learning_rate = opt.learning_rate * (i / opt.warmup_steps)
            else:
                learning_rate = opt.learning_rate

            learning_rate = learning_rate * (opt.decay_rate ** (
                i / opt.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate

            if model_depth == 0:
                z = z['image'].to(device)
            else:
                z = z.to(device)
            # reconstruction of current feature
            z_hat = model.forward_step(z, model_depth)
            loss = criterion(z_hat, z)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= max_samples

        if opt.wandb:
            wandb.log({'loss': total_loss}, step=epoch)

        print("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
                                                     datetime.timedelta(seconds=time.time() - start)))

    # at the end of each training cycle, convert the training data to the compressed representation through
    # the currently trained encoder

    # we want to replace the dataloader with the compressed data
    with torch.no_grad():
        model.eval()
        z_new = []
        idx = 0
        for z in train_dataloader:
            if idx >= max_samples:
                break
            idx += 1
            if model_depth == 0:
                z = z['image'].to(device)
            else:
                z = z.to(device)
            z_fwd = model.get_compressed(z, model_depth).detach().clone()
            z_new.append(z_fwd)

        z_new = torch.cat(z_new, dim=0).cpu()

    # train_set = torch.utils.data.TensorDataset(z_new)
    train_dataloader = torch.utils.data.DataLoader(z_new, batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=opt.num_workers)
    # if model_depth == 0:
    #     for param in model.slot_attention_autoencoder.parameters():
    #         param.requires_grad = False
    # else:
    #     for param in model.token_compressor[model_depth-1].parameters():
    #         param.requires_grad = False

    # break
    if opt.wandb:
        wandb.finish()

    torch.save({
        'model_state_dict': model.state_dict(),
    }, opt.results_dir + f"/{model_filename}.ckpt")
