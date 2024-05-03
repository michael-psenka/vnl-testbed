import torch.nn as nn
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

import geoopt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, value, key, query):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = value.reshape(N, value_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix mul & sum for attention scores
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = attention / (self.embed_size ** (1 / 2))
        attention = torch.softmax(attention, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class SlotTransformer(nn.Module):
    def __init__(self, num_slots, dim, iters=3, heads=8, dropout=0.1, forward_expansion=4, use_transformer_encoder=False, use_transformer_decoder=False, max_seq_length=512):
        super(SlotTransformer, self).__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.use_transformer_encoder = use_transformer_encoder
        self.use_transformer_decoder = use_transformer_decoder

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        if self.use_transformer_encoder:
            self.token_encoder = TokenEncoder(
                emb_size=dim, depth=iters, heads=heads, mlp_dim=forward_expansion * dim, max_seq_length=max_seq_length)
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        dim,
                        heads,
                        dropout=dropout,
                        forward_expansion=forward_expansion,
                    )
                    for _ in range(iters)
                ]
            )

        if self.use_transformer_decoder:
            self.token_decoder = TokenDecoder(
                emb_size=dim, depth=iters, heads=heads, mlp_dim=forward_expansion * dim, max_seq_length=max_seq_length)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        if self.use_transformer_encoder:
            slots = self.token_encoder(slots)
        else:
            for transformer in self.transformer_blocks:
                slots = transformer(inputs, inputs, slots)

        if self.use_transformer_decoder:
            slots = self.token_decoder(slots)

        return slots

# standard transformer encoder/decoder using torch.nn


class TokenEncoder(nn.Module):
    def __init__(self, emb_size=256, depth=4, heads=8, mlp_dim=512, max_seq_length=512):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_length, emb_size))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=heads, dim_feedforward=mlp_dim, activation="relu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        seq_length = x.size(1)
        x = x + self.pos_embedding[:, :seq_length, :]
        return self.transformer(x)


class TokenDecoder(nn.Module):
    def __init__(self, emb_size=256, depth=4, heads=8, mlp_dim=512, max_seq_length=512):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_length, emb_size))
        layer = nn.TransformerDecoderLayer(
            d_model=emb_size, nhead=heads, dim_feedforward=mlp_dim, activation="relu")
        self.transformer = nn.TransformerDecoder(layer, num_layers=depth)

    def forward(self, z):
        seq_length = z.size(1)
        z = z + self.pos_embedding[:, :seq_length, :]
        # In a typical autoencoder, the memory and target could be the same. However, for simplicity, we're just using the encoded representation directly.
        # An additional token for the start of sequence might be added, and similarly for the target during training.
        memory = z.detach().clone()
        return self.transformer(z, memory)


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)


"""Adds soft positional embedding with learnable projection."""


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class EncoderCNN(nn.Module):
    def __init__(self, resolution, hid_dim, depth=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, padding=2)
        self.layers = nn.ModuleList(
            [nn.Conv2d(hid_dim, hid_dim, 5, padding=2) for i in range(depth-1)])
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        for layer in self.layers:
            x = F.relu(layer(x))
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x


class DecoderCNN(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(
            2, 2), padding=2, output_padding=1).to(device)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(
            2, 2), padding=2, output_padding=1).to(device)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(
            2, 2), padding=2, output_padding=1).to(device)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(
            2, 2), padding=2, output_padding=1).to(device)
        self.conv5 = nn.ConvTranspose2d(
            hid_dim, hid_dim, 5, stride=(1, 1), padding=2).to(device)
        self.conv6 = nn.ConvTranspose2d(
            hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(
            hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:, :, :self.resolution[0], :self.resolution[1]]
        x = x.permute(0, 2, 3, 1)
        return x


"""Slot Attention-based auto-encoder for object discovery."""


class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim, cnn_depth=4, use_trfmr=False, use_transformer_encoder=False, use_transformer_decoder=False, pre_trained_model=None):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.cnn_depth = cnn_depth

        self.encoder = EncoderCNN(
            self.resolution, self.hid_dim, depth=self.cnn_depth)
        self.decoder = DecoderCNN(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        if use_trfmr:
            self.slot_attention = SlotTransformer(
                num_slots=self.num_slots,
                dim=hid_dim,
                use_transformer_encoder=use_transformer_encoder,
                use_transformer_decoder=use_transformer_decoder,
            )
        else:
            self.slot_attention = SlotAttention(
                num_slots=self.num_slots,
                dim=hid_dim,
                iters=self.num_iterations,
                eps=1e-8,
                hidden_dim=128)

    def encode(self, image):
        x = self.encoder(image)
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.

        return x

    def slot_att(self, x, num_slots=None):
        slots = self.slot_attention(x, num_slots)
        return slots

    def slot_merge(self, slots, n_clusters=2):
        if not isinstance(slots, np.ndarray):
            slots = np.array(slots.cpu().detach())

        cluster_labels = AgglomerativeClustering(
            n_clusters=n_clusters).fit_predict(slots)
        cluster_means = np.array(
            [slots[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])
        cluster_means = torch.tensor(cluster_means).to(device)

        return cluster_means

    def decode(self, slots, batch_size):
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))

        x = self.decoder(slots)
        recons, masks = x.reshape(
            batch_size, -1, x.shape[1], x.shape[2], x.shape[3]).split([3, 1], dim=-1)
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)
        recon_combined = recon_combined.permute(0, 3, 1, 2)

        return recon_combined, recons, masks, slots

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder(image)
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))

        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(
            image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3, 1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots

class TokenCompressor(nn.Module):
    def __init__(self, emb_size=256, depth=4, heads=8, mlp_dim=512, max_seq_length=7):
        super().__init__()
        self.encoder = TokenEncoder(
            emb_size=emb_size, depth=depth, heads=heads, mlp_dim=mlp_dim, max_seq_length=max_seq_length)
        

    def forward(self, x):
        # encode
        z = self.encoder(x)[:,:-1,:]
        return z

class TokenCompressionAutoencoder(nn.Module):
    def __init__(self, emb_size=256, depth=4, heads=8, mlp_dim=512, max_seq_length=7):
        super().__init__()
        self.encoder = TokenEncoder(
            emb_size=emb_size, depth=depth, heads=heads, mlp_dim=mlp_dim, max_seq_length=max_seq_length)
        self.decoder = TokenDecoder(
            emb_size=emb_size, depth=depth, heads=heads, mlp_dim=mlp_dim, max_seq_length=max_seq_length)
        # trainable token to append back to sequence
        self.trainable_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x):
        # encode
        z = self.encoder(x)
        # remove last token and replace it with the trainable token. output of shape (B, emb_size)
        new_slots = self.trainable_token.repeat(z.shape[0], 1, 1).squeeze(1)
        z[:, -1, :] = new_slots
        # now decode and add skip connection
        z = self.decoder(z)  # + x
        return z

    def encode(self, x):
        return self.encoder(x)[:, :-1, :]

    def decode(self, z):
        z = torch.cat(
            [z, self.trainable_token.repeat(z.shape[0], 1, 1)], dim=1)
        return self.decoder(z)


# full model. first runs normal slot attention, then trains downstream token compressor models

class SlotAttentionCompressionAutoencoder(nn.Module):
    def __init__(self, slot_attention_autoencoder: SlotAttentionAutoEncoder, num_slots: int = 7, hid_dim: int = 256):

        super().__init__()
        self.slot_attention_autoencoder = slot_attention_autoencoder
        # list of token compressors, each reduces number of tokens by 1
        self.token_compressor = nn.ModuleList([TokenCompressionAutoencoder(
            emb_size=hid_dim, depth=4, heads=8, mlp_dim=512, max_seq_length=num_slots) for _ in range(num_slots-1)])

    # this will go through the entire, full antoencoding. note this is rarely used in training,
    # see forward_step

    def forward(self, x, upto: int = None):
        # print('here')
        # recon_combined, recons, masks, slots = self.slot_attention_autoencoder(
        #     x)
        if upto is None:
            upto = len(self.token_compressor)
        # first run the encoding from slot attention
        z = self.slot_attention_autoencoder.encode(x)
        z = self.slot_attention_autoencoder.slot_att(z)

        # run through token compressors
        # for i in range(len(self.token_compressor)):
        for i in range(upto):
            z = self.token_compressor[i].encode(z)

        # now run through decoders
        # for i in range(len(self.token_compressor)-1, -1, -1):
        for i in range(upto-1, -1, -1):
            z = self.token_compressor[i].decode(z)

        recon_combined, recons, masks, slots = self.slot_attention_autoencoder.decode(
            z, x.shape[0])

        return recon_combined, recons, masks, slots

    # forward_step will be mostly used during training. Only runs the encoding/decoding process
    # at a certain step. Expects input from previous step. For example, at step 0, input is
    # original data just runs normal slot attention. at step 1, input is the slot attention
    # features, and output is reconstruction for the first token compressor

    def forward_step(self, z, step: int):
        if step == 0:
            x_hat, _, _, _ = self.slot_attention_autoencoder(z)
            return x_hat

        else:
            x_hat = self.token_compressor[step-1](z)
            return x_hat

    # Used for inference when we want the current fowarded representations to be the next data input
    # Notice how we encode and decode the token compressor since we want to get the output representations

    def get_compressed(self, z, step: int):
        if step == 0:
            z = self.slot_attention_autoencoder.encode(z)
            return self.slot_attention_autoencoder.slot_att(z)
        else:
            return self.token_compressor[step-1](z)

    # encode_step will do a single step of the encoding. for example, if step 0, will run
    # normal slot attention encoding. if step 1, expects slot attention features, returns
    # encoding through first token compressor
    # Notice how we encode and decode the token compressor since we want to get the output representations

    def encode_step(self, z, step: int):
        if step == 0:
            z = self.slot_attention_autoencoder.encode(z)
            return self.slot_attention_autoencoder.slot_att(z)
        else:
            return self.token_compressor[step-1].encode(z)

    # encode_full will return a list of features at all fidelities. Note this is rarely
    # if ever used in actual training
    def encode_full(self, z):

        features_full = []

        # run through token compressors
        for i in range(self.num_slots):
            z = self.encode_step(z, i)
            features_full.append(z.detach().clone())

        return features_full


# NEW FULL MODEL: now reconstruction goes all the way back to the original data
class SlotAttentionCompressionAutoencoderDirect(nn.Module):
    def __init__(self, slot_attention_autoencoder: SlotAttentionAutoEncoder, num_slots: int = 7, hid_dim: int = 256):

        super().__init__()
        # right now, there is only one decoder throughout the model. this can be changed
        self.slot_attention_autoencoder = slot_attention_autoencoder
        # list of token compressors, each reduces number of tokens by 1
        self.token_compressor = nn.ModuleList([TokenCompressor(
            emb_size=hid_dim, depth=4, heads=8, mlp_dim=512, max_seq_length=num_slots) for _ in range(num_slots-1)])
        
        

    # this will go through the entire, full antoencoding. note this is rarely used in training,
    # see forward_step

    def forward(self, x, upto: int = None):
        # print('here')
        # recon_combined, recons, masks, slots = self.slot_attention_autoencoder(
        #     x)
        if upto is None:
            upto = len(self.token_compressor)
        # first run the encoding from slot attention
        z = self.slot_attention_autoencoder.encode(x)
        z = self.slot_attention_autoencoder.slot_att(z)

        # run through token compressors
        # for i in range(len(self.token_compressor)):
        for i in range(upto):
            z = self.token_compressor[i](z)

        # decode now goes straight from tokens to image

        recon_combined, recons, masks, slots = self.slot_attention_autoencoder.decode(
            z, x.shape[0])

        return recon_combined, recons, masks, slots

    # forward_step will be mostly used during training. Only runs the encoding/decoding process
    # at a certain step. Expects input from previous step. For example, at step 0, input is
    # original data just runs normal slot attention. at step 1, input is the slot attention
    # features, and output is reconstruction for the first token compressor

    def forward_step(self, z, step: int):
        if step == 0:
            x_hat, _, _, _ = self.slot_attention_autoencoder(z)
            return x_hat

        else:
            z_km1 = self.token_compressor[step-1](z)
            x_hat, _, _, _ = self.slot_attention_autoencoder.decode(z_km1, z.shape[0])
            return x_hat

    # Used for inference when we want the current fowarded representations to be the next data input
    # Notice how we encode and decode the token compressor since we want to get the output representations

    def get_compressed(self, z, step: int):
        if step == 0:
            z = self.slot_attention_autoencoder.encode(z)
            return self.slot_attention_autoencoder.slot_att(z)
        else:
            z = self.token_compressor[step-1](z)
            x_hat, _, _, _ = self.slot_attention_autoencoder.decode(z, z.shape[0])
            return x_hat

    # encode_step will do a single step of the encoding. for example, if step 0, will run
    # normal slot attention encoding. if step 1, expects slot attention features, returns
    # encoding through first token compressor
    # Notice how we encode and decode the token compressor since we want to get the output representations

    def encode_step(self, z, step: int):
        if step == 0:
            z = self.slot_attention_autoencoder.encode(z)
            return self.slot_attention_autoencoder.slot_att(z)
        else:
            return self.token_compressor[step-1](z)

    # encode_full will return a list of features at all fidelities. Note this is rarely
    # if ever used in actual training
    def encode_full(self, z):

        features_full = []

        # run through token compressors
        for i in range(self.num_slots):
            z = self.encode_step(z, i)
            features_full.append(z.detach().clone())

        return features_full
