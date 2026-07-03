"""
Localizer MRI Encoder - Vision Transformer for 3 consecutive localizer scout images
localizer-specific processing.

Input: 3 consecutive 2D localizer images [batch, 3, 224, 224]
Output: Embedding tensor [batch, embed_dim]

The 3 localizer images are from oblique-axial plane 
"""

from functools import partial
from typing import Literal
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Block
from timm.models.layers import PatchEmbed


class MultiViewAggregation(nn.Module):
    """
    Aggregate embeddings from multiple views.
    Methods: 'mean', 'concat', 'attention'
    """
    def __init__(self, embed_dim, num_views=3, method='mean'):
        super().__init__()
        self.method = method
        self.num_views = num_views
        
        if method == 'attention':
            self.attention_weights = nn.Parameter(torch.ones(num_views) / num_views)
        elif method == 'concat':
         
            self.projection = nn.Linear(embed_dim * num_views, embed_dim)
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: List of [batch, embed_dim] tensors, one per view
        Returns:
            [batch, embed_dim] aggregated embedding
        """
        if self.method == 'mean':
            return torch.stack(embeddings, dim=0).mean(dim=0)
        
        elif self.method == 'attention':
            # Softmax attention weights
            weights = torch.softmax(self.attention_weights, dim=0)
            # Weighted sum: sum_i (weight_i * embedding_i)
            weighted = torch.stack([w * emb for w, emb in zip(weights, embeddings)], dim=0)
            return weighted.sum(dim=0)
        
        elif self.method == 'concat':
            # Concatenate and project
            concatenated = torch.cat(embeddings, dim=-1)  # [batch, embed_dim * num_views]
            return self.projection(concatenated)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")


class LocalizerEncoder(VisionTransformer):
    """
    Vision Transformer for localizer scout images.
    
    Two modes:
    1. Single-pass: Treat 3 localizer images as 3 channels (like RGB)
    2. Multi-view: Process each image independently, then aggregate
    """
    
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3,
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 global_pool=True,
                 multi_view_mode=False,
                 aggregation_method='mean',
                 **kwargs):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size for ViT
            in_chans: Number of input channels (3 for localizer)
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            global_pool: If True, use global average pooling; else CLS token
            multi_view_mode: If True, process 3 images independently and aggregate
            aggregation_method: How to aggregate multi-view embeddings ('mean', 'attention', 'concat')
        """
        super().__init__(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans,
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads,
            mlp_ratio=mlp_ratio, 
            **kwargs
        )
        
        self.global_pool = global_pool
        self.multi_view_mode = multi_view_mode
        
        if self.global_pool:
            self.fc_norm = nn.LayerNorm(embed_dim)
            del self.norm  # Remove original norm
        
        if multi_view_mode:
            
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=1,  # Process each localizer image separately
                embed_dim=embed_dim
            )
            
            # Aggregation module
            self.aggregation = MultiViewAggregation(
                embed_dim=embed_dim,
                num_views=3,
                method=aggregation_method
            )
    
    def forward_single_pass(self, x):
        """
        Process all 3 localizer images together as 3-channel input.
        
        Args:
            x: [batch, 3, 224, 224] tensor
        Returns:
            [batch, embed_dim] embedding
        """
        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        if self.global_pool:
            x = x[:, 1:].mean(dim=1)  # Global pool without CLS token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]  # CLS token
        
        return outcome
    
    def forward_multi_view(self, x):
        """
        Process each of the 3 localizer images independently, then aggregate.
        
        Args:
            x: [batch, 3, 224, 224] tensor
        Returns:
            [batch, embed_dim] aggregated embedding
        """
        batch_size = x.shape[0]
        embeddings = []
        
        # Process each view independently
        for i in range(3):
            view = x[:, i:i+1, :, :]  # [batch, 1, 224, 224]
            
            # Patch embedding
            view_patches = self.patch_embed(view)
            view_patches = self._pos_embed(view_patches)
            view_patches = self.norm_pre(view_patches)
            
            # Transformer blocks
            for blk in self.blocks:
                view_patches = blk(view_patches)
            
            # Extract embedding
            if self.global_pool:
                view_emb = view_patches[:, 1:].mean(dim=1)
                view_emb = self.fc_norm(view_emb)
            else:
                view_emb = self.norm(view_patches)[:, 0]
            
            embeddings.append(view_emb)
        
        # Aggregate embeddings from all views
        aggregated = self.aggregation(embeddings)
        return aggregated
    
    def forward_features(self, x):
        """
        Main forward pass 
        
        Args:
            x: [batch, 3, 224, 224] tensor
        Returns:
            [batch, embed_dim] embedding
        """
        if self.multi_view_mode:
            return self.forward_multi_view(x)
        else:
            return self.forward_single_pass(x)
    
    def forward(self, x):
        """
        Args:
            x: [batch, 3, 224, 224] localizer tensor
        Returns:
            [batch, embed_dim] embedding
        """
        return self.forward_features(x)


# model variants


def vit_tiny_localizer(multi_view_mode=False, aggregation_method='mean', **kwargs):
    """Tiny model: 384 dim, 3 layers"""
    model = LocalizerEncoder(
        patch_size=16, 
        embed_dim=384, 
        depth=3, 
        num_heads=6, 
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        multi_view_mode=multi_view_mode,
        aggregation_method=aggregation_method,
        **kwargs
    )
    return model


def vit_small_localizer(multi_view_mode=False, aggregation_method='mean', **kwargs):
    """Small model: 512 dim, 4 layers"""
    model = LocalizerEncoder(
        patch_size=16, 
        embed_dim=512, 
        depth=4, 
        num_heads=8, 
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        multi_view_mode=multi_view_mode,
        aggregation_method=aggregation_method,
        **kwargs
    )
    return model


def vit_base_localizer(multi_view_mode=False, aggregation_method='mean', **kwargs):
    """
    Base model: 768 dim, 12 layers

    """
    model = LocalizerEncoder(
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        multi_view_mode=multi_view_mode,
        aggregation_method=aggregation_method,
        **kwargs
    )
    return model



# Utility function to build localizer encoder


def build_localizer_encoder(model_name='vit_base_localizer',
                            global_pool=True,
                            multi_view_mode=False,
                            aggregation_method='mean',
                            pretrained_path=None,
                            **kwargs):
    """
    Build localizer encoder with optional pretrained weights.
    
    Args:
        model_name: One of ['vit_tiny_localizer', 'vit_small_localizer', 'vit_base_localizer']
        global_pool: If True, use global average pooling; else use CLS token
        multi_view_mode: If True, process 3 images independently and aggregate
        aggregation_method: 'mean', 'attention', or 'concat'
        pretrained_path: Path to pretrained weights (optional)
        **kwargs: Additional arguments for model
        
    Returns:
        LocalizerEncoder model
    """
    # Get model builder function
    model_fn = globals().get(model_name)
    if model_fn is None:
        raise ValueError(f"Unknown localizer model: {model_name}")
    
    # Build model
    model = model_fn(
        global_pool=global_pool,
        multi_view_mode=multi_view_mode,
        aggregation_method=aggregation_method,
        **kwargs
    )
    
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
        print(f"Loaded localizer pretrained weights from {pretrained_path}")
        if msg.missing_keys:
            print(f"Missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"Unexpected keys: {msg.unexpected_keys}")
    
    return model

