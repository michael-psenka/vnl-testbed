import torch
import torch.nn as nn


# standard transformer encoder/decoder using torch.nn


class TokenEncoder(nn.Module):
    def __init__(self, emb_size=256, depth=4, heads=8, mlp_dim=512, max_seq_length=6):
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
    def __init__(self, emb_size=256, depth=4, heads=8, mlp_dim=512, max_seq_length=7):
        super().__init__()
        # Add 1 to max_seq_length to account for the additional slot
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_length+1, emb_size))
        layer = nn.TransformerDecoderLayer(
            d_model=emb_size, nhead=heads, dim_feedforward=mlp_dim, activation="relu")
        self.transformer = nn.TransformerDecoder(layer, num_layers=depth)
        self.additional_slot = nn.Parameter(
            torch.randn(1, 1, emb_size))  # Hardcoded, add one slot

    def forward(self, z):
        new_slots = self.additional_slot.repeat(z.shape[0], 1, 1)
        z = torch.cat([z, new_slots], dim=1)
        seq_length = z.size(1)
        z = z + self.pos_embedding[:, :seq_length, :]
        # In a typical autoencoder, the memory and target could be the same. However, for simplicity, we're just using the encoded representation directly.
        # An additional token for the start of sequence might be added, and similarly for the target during training.
        memory = z.detach().clone()
        return self.transformer(z, memory)


class Transformer(nn.Module):
    def __init__(self, emb_size=256, depth=4, heads=8, mlp_dim=512, max_seq_length=6):
        super().__init__()
        self.encoder = TokenEncoder(
            emb_size=emb_size, depth=depth, heads=heads, mlp_dim=mlp_dim, max_seq_length=max_seq_length)
        self.decoder = TokenDecoder(
            emb_size=emb_size, depth=depth, heads=heads, mlp_dim=mlp_dim, max_seq_length=max_seq_length)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
