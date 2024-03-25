import torch

# Load the state_dict
weights = torch.load(
    '/shared/rzhang/slot_att/results/objects-all-slots-7.ckpt')['model_state_dict']

# Create a new state_dict with updated keys
updated_weights = {}

for key in weights.keys():
    # Renaming encoder_cnn and decoder_cnn to encoder and decoder respectively
    new_key = key.replace('encoder_cnn', 'encoder').replace(
        'decoder_cnn', 'decoder')
    updated_weights[new_key] = weights[key]

# Optionally, print to verify the key names have been updated
for key in updated_weights.keys():
    print(key)

torch.save({
    'model_state_dict': updated_weights
}, '/shared/rzhang/slot_att/results/objects-all-slots-7.ckpt')
