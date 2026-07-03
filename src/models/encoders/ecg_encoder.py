"""
ECG Encoder - Vision Transformer for 12-lead ECG signals
Uses 1D temporal patching to preserve the temporal structure of ECG data.

Input: ECG tensor [batch, 12, 5000]
Output: Embedding tensor [batch, embed_dim]
"""

from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer


class ECG1DPatchEmbed(nn.Module):
    """
    1D Patch embedding for ECG signals.
    Converts [batch, 12, 5000] -> [batch, num_patches, embed_dim]
    
Create patches along the time dimension
    Input: [batch, 12, 5000] (12 leads, 5000 time points)
    Patch along time: Each patch = [12, patch_size] (all leads, temporal window)
    Example with patch_size=100: 5000/100 = 50 patches
    Each patch flattened: [12, 100] -> [1200]
    Linear projection: [1200] -> [embed_dim]
    Output: [batch, 50, embed_dim]
    
    """
    def __init__(self, signal_length=5000, num_leads=12, patch_size=100, embed_dim=768):
        super().__init__()
        self.signal_length = signal_length
        self.num_leads = num_leads
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        #  number of patches
        self.num_patches = signal_length // patch_size
        
        # Linear projection from flattened patch to embedding
        # each patch: [num_leads, patch_size] -> [num_leads * patch_size]
        self.projection = nn.Linear(num_leads * patch_size, embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: [batch, 12, 5000] ECG tensor
        Returns:
            [batch, num_patches, embed_dim]
        """
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        B, C, L = x.shape
        assert C == self.num_leads, f"Expected {self.num_leads} leads, got {C}"
        
        # Truncate or pad to exact multiple of patch_size
        target_length = self.num_patches * self.patch_size
        if L < target_length:
            x = torch.nn.functional.pad(x, (0, target_length - L))
        elif L > target_length:
            x = x[:, :, :target_length]
        
        # Reshape into patches: [B, 12, 5000] -> [B, 50, 12, 100]
        #  group consecutive time points into patches
        x = x.reshape(B, C, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # [B, num_patches, 12, patch_size]
        
        # flatten each patch: [B, num_patches, 12, patch_size] -> [B, num_patches, 12*patch_size]
        x = x.reshape(B, self.num_patches, C * self.patch_size)
        
        # Project to embedding dimension
        x = self.projection(x)  # [B, num_patches, embed_dim]
        
        return x


class ECGEncoder(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer encoder for ECG signals using 1D temporal patching.
    Supports global average pooling or CLS token extraction.
    """
    def __init__(self, global_pool=False, signal_length=5000, num_leads=12, patch_size=100, **kwargs):
   
        if 'patch_size' not in kwargs:
            kwargs['patch_size'] = 16  # Dummy value for parent init
        
        # Initialize parent ViT with minimal config
        super(ECGEncoder, self).__init__(**kwargs)
        
        
        self.patch_embed = ECG1DPatchEmbed(
            signal_length=signal_length,
            num_leads=num_leads,
            patch_size=patch_size,
            embed_dim=kwargs['embed_dim']
        )
        
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, kwargs['embed_dim'])
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.global_pool = global_pool
        if self.global_pool == "attention_pool":
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=kwargs['embed_dim'], 
                num_heads=kwargs['num_heads'], 
                batch_first=True
            )
        if self.global_pool:
            norm_layer = kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6))
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

    def forward_features(self, x, localized=False):
        """
        Forward pass through encoder.
        
        Args:
            x: [batch, 12, 5000] ECG tensor
            localized: If True, return all patch tokens without pooling
            
        Returns:
            [batch, embed_dim] or [batch, num_patches, embed_dim] if localized
        """
        B = x.shape[0]
        
        # Patch embedding: [B, 12, 5000] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Output handling
        if localized:
            # Return all patch tokens (no CLS)
            outcome = x[:, 1:]
        elif self.global_pool == "attention_pool":
            # Attention pooling over patch tokens
            q = x[:, 1:, :].mean(dim=1, keepdim=True)
            k = x[:, 1:, :]
            v = x[:, 1:, :]
            x, _ = self.attention_pool(q, k, v)
            outcome = self.fc_norm(x.squeeze(dim=1))
        elif self.global_pool:
            # Global average pooling
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            # Use CLS token
            x = self.norm(x)
            outcome = x[:, 0]
        
        return outcome
    
    def forward(self, x):
        """
        Args:
            x: [batch, 12, 5000] ECG tensor
        Returns:
            [batch, embed_dim] embedding
        """
        return self.forward_features(x)



# model variants 


def vit_pluto_patchX(patch_size=100, **kwargs):
    """
    Smallest model: 256 dim, 3 layers
    Default patch_size=100 gives 50 patches from 5000 samples
    """
    model = ECGEncoder(
        patch_size=patch_size,
        embed_dim=256, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def vit_tiny_patchX(patch_size=100, **kwargs):
    """
    Tiny model: 384 dim, 3 layers

    """
    model = ECGEncoder(
        patch_size=patch_size,
        embed_dim=384, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def vit_small_patchX(patch_size=100, **kwargs):
    """
    Small model: 512 dim, 4 layers

    """
    model = ECGEncoder(
        patch_size=patch_size,
        embed_dim=512, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def vit_medium_patchX(patch_size=100, **kwargs):
    """
    Medium model: 640 dim, 6 layers

    """
    model = ECGEncoder(
        patch_size=patch_size,
        embed_dim=640, depth=6, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def vit_base_patchX(patch_size=100, **kwargs):
    """
    Base model: 768 dim, 12 layers

    """
    model = ECGEncoder(
        patch_size=patch_size,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


# Utility function to build ECG encoder


def build_ecg_encoder(model_name='vit_tiny_patchX', 
                      global_pool=True,
                      patch_size=100,
                      pretrained_path=None,
                      **kwargs):
    """
    Build ECG encoder with optional pretrained weights.
    
    Args:
        model_name: One of ['vit_pluto_patchX', 'vit_tiny_patchX', 'vit_small_patchX', ...]
        global_pool: If True, use global average pooling; else use CLS token
        patch_size: Size of temporal patches (default 100 -> 50 patches from 5000 samples)
                   Options: 50 (100 patches), 100 (50 patches), 125 (40 patches), 250 (20 patches)
        pretrained_path: Path to pretrained weights (optional)
        **kwargs: Additional arguments for model
        
    Returns:
        ECGEncoder model
    """
    # Get model builder function
    model_fn = globals().get(model_name)
    if model_fn is None:
        raise ValueError(f"Unknown ECG model: {model_name}")
    
    # Build model
    model = model_fn(global_pool=global_pool, patch_size=patch_size, **kwargs)
    
    # Load pretrained weights if provided
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load with strict=False to allow partial loading
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded ECG pretrained weights from {pretrained_path}")
        if msg.missing_keys:
            print(f"Missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"Unexpected keys: {msg.unexpected_keys}")
    
    return model
