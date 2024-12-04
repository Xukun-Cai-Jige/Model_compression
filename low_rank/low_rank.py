import os
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn

import json

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LowRankLinear, self).__init__()
        self.rank = max(1, rank)
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear1 = nn.Linear(in_features, self.rank, bias=False)
        self.linear2 = nn.Linear(self.rank, out_features, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    @property
    def weight(self):
        # Mimic the weight attribute by combining the two linear layers
        # FIXME Come up with a better way to handle this
        return self.linear2.weight.data @ self.linear1.weight.data
    
    @classmethod
    def from_linear(cls, linear_layer, retained_variance):
        """
        Initialize from an existing nn.Linear layer using SVD decomposition
        with proper dimension handling
        """
        device = linear_layer.weight.device
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        # Init low rank layers using SVD
        with torch.no_grad():
            try:
                U, S, V = torch.svd(linear_layer.weight.data)
                total_variance = torch.sum(S ** 2)
                cumulative_variance = torch.cumsum(S ** 2, dim=0)
                rank = torch.searchsorted(cumulative_variance, retained_variance * total_variance).item()
                # Ensure minimum rank of 1
                # init instance
                instance = cls(in_features, out_features, rank)
                instance = instance.to(device)
                rank = instance.rank
            
                # Expected dimensions:
                # linear1.weight -> (rank, in_features)
                # linear2.weight -> (out_features, rank)
                sqrt_s = torch.sqrt(S[:rank]).view(-1, 1)
                instance.linear1.weight.data = (V[:, :rank] * sqrt_s.T).T
                instance.linear2.weight.data = U[:, :rank] * sqrt_s.T
                
                # Handle bias at 2nd layer
                if linear_layer.bias is not None:
                    instance.linear2.bias.data.copy_(linear_layer.bias.data)
                    
                # Sanity check
                assert instance.linear1.weight.shape == (rank, in_features), \
                    f"Linear1 weight shape mismatch: got {instance.linear1.weight.shape}, expected {(rank, in_features)}"
                assert instance.linear2.weight.shape == (out_features, rank), \
                    f"Linear2 weight shape mismatch: got {instance.linear2.weight.shape}, expected {(out_features, rank)}"
                
            except Exception as e:
                raise RuntimeError(f"SVD initialization failed: {str(e)}\n"
                                 f"Shapes: weight={linear_layer.weight.shape}, "
                                 f"U={U.shape}, S={S.shape}, V={V.shape}, "
                                 f"rank={rank}")
            
        return instance
    
    def get_compression_stats(self):
        """
        Return compression statistics
        """
        original_params   = self.in_features * self.out_features + self.out_features
        compressed_params = (self.in_features * self.rank +    # first layer weights
                             self.rank * self.out_features +   # second layer weights
                             self.out_features)                # bias
        
        stats = {
            'original_params': original_params,
            'compressed_params': compressed_params,
            'compression_ratio': compressed_params / original_params,
            'rank': self.rank,
        }
        return stats


metadata = {}
def replace_linear_with_low_rank(model, retained_variance, skip_patterns=None):
    """
    Replace linear layers with low-rank versions initialized using SVD
    """
    if skip_patterns is None:
        skip_patterns = []

    replacements = 0
    total_params_before = 0
    total_params_after = 0
    
    for name, module in model.named_modules():
        if any(pattern in name for pattern in skip_patterns):
            continue
            
        if isinstance(module, nn.Linear):
            try:
                print(f"\nProcessing layer {name}")
                print(f"Original shape: {module.weight.shape}")
                
                low_rank_module = LowRankLinear.from_linear(module, retained_variance)                
                print(f"Low rank shapes: {low_rank_module.linear1.weight.shape} -> {low_rank_module.linear2.weight.shape}")
                
                # Compression stats
                stats = low_rank_module.get_compression_stats()
                total_params_before += stats['original_params']
                total_params_after += stats['compressed_params']
                
                # Replace the module
                parent_module = model
                components = name.split('.')
                for comp in components[:-1]:
                    parent_module = getattr(parent_module, comp)
                setattr(parent_module, components[-1], low_rank_module)

                # Save the metadata for reconstruction later
                metadata[name] = {
                    "retained_variance": retained_variance,
                    "original_shape": list(module.weight.shape),
                    "low_rank_shape": {
                        "linear1": list(low_rank_module.linear1.weight.shape),
                        "linear2": list(low_rank_module.linear2.weight.shape)
                    }
                }
                
                replacements += 1
                
                # Print retained variance
                with torch.no_grad():
                    U, S, V = torch.svd(module.weight.data)
                    rank = low_rank_module.rank
                    total_variance = torch.sum(S ** 2)
                    module_retained_var = torch.sum(S[:rank] ** 2)
                    variance_ratio = module_retained_var / total_variance
                    print(f"Layer {name}: Retaining {variance_ratio:.2%} of variance")
                    
            except Exception as e:
                print(f"Warning: Could not replace layer {name}: {e}")
                continue
            
    compression_ratio = total_params_after / total_params_before
    print(f"\nSummary:")
    print(f"Replaced {replacements} linear layers")
    print(f"Parameters before: {total_params_before:,}")
    print(f"Parameters after: {total_params_after:,}")
    print(f"Compression ratio: {compression_ratio:.2%}")
    
    return model

def get_metadata():
    return metadata

def save_metadata(path):
    with open(path, "w") as f:
        json.dump(metadata, f,  indent=2)

def read_metadata(path):
    with open(path, "r") as f:
        metadata = json.load(f)
    return metadata

def clear_metadata():
    global metadata
    metadata = {}

def patch_model_using_metadata(model, metadata, pt_model_path=None):
    """
    Replace linear layers with low-rank layers using metadata.
    """
    for name, layer_info in metadata.items():
        # Locate the module to replace
        parent_module = model
        components = name.split('.')
        for comp in components[:-1]:
            parent_module = getattr(parent_module, comp)

        original_layer = getattr(parent_module, components[-1])
        if isinstance(original_layer, torch.nn.Linear):
            # Replace with a LowRankLinear equivalent using metadata
            in_features = layer_info["low_rank_shape"]["linear1"][1]
            out_features = layer_info["low_rank_shape"]["linear2"][0]
            rank = layer_info["low_rank_shape"]["linear1"][0]
            low_rank_layer = LowRankLinear(in_features, out_features, rank)
            setattr(parent_module, components[-1], low_rank_layer)

    if pt_model_path is not None:
        loc = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(pt_model_path, map_location=loc)
        model.load_state_dict(state_dict)
        print('Loaded model weight on ' + loc)

    return model
