import torch

# Load the state_dict
weights = torch.load(
    '/shared/rzhang/slot_att/results/objects-all-slots-7.ckpt')['model_state_dict']

# Create a new state_dict with updated keys
updated_weights = {}

for key, value in weights.items():
    # Identify and rename encoder_cnn and decoder_cnn layers
    if 'encoder.conv' in key:
        # Extract layer number and adjust it to match the layers structure
        # Subtract 1 to adjust indexing if necessary
        layer_num = int(key.split('.')[1][-1]) - 1
        new_key = key.replace(
            f'encoder.conv{layer_num + 1}', f'encoder.layers.{layer_num}')
    elif 'decoder' in key:
        # Adjust according to your needs
        new_key = key.replace('decoder', 'decoder')
    else:
        new_key = key  # No change for other keys

    updated_weights[new_key] = value

# Verify the key names have been updated correctly
for key in updated_weights.keys():
    print(key)

# Save the updated state_dict back to a checkpoint
torch.save({
    'model_state_dict': updated_weights
}, '/shared/rzhang/slot_att/results/objects-all-slots-7.ckpt')
