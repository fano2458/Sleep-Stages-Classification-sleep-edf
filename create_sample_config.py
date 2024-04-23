import json

# Define the model configuration dictionary
config = {
  "kernel_sizes": [(1, 10), (1, 10), (1, 10)],
  "stride": (1, 1),
  "padding": (0, 0),
  "pooling_kernel_size": (1, 4),
  "pooling_stride": (1, 4),
  "pooling_padding": (0, 0),
  "n_conv_layers": 3,
  "n_input_channels": 1,
  "in_planes": 32,
  "activation": "ReLU",
  "max_pool": True,
  "conv_bias": False,
  "dim": 32,
  "num_layers": 3,
  "num_heads": 2,
  "num_classes": 5,
  "attn_dropout": 0.2,
  "dropout": 0.2,
  "mlp_size": 64,
  "positional_emb": "learnable"
}

# Save the configuration to a JSON file
with open('cct_config.json', 'w') as jsonfile:
  json.dump(config, jsonfile, indent=4)
