"""
Tri-Modal CLIP - Contrastive learning model for Localizer + ECG + Tabular data
Following Synergy-CLIP architecture for multi-modal alignment.

shared temperature (like Synergy-CLIP/ViTa)
Configurable loss weights (α, β, γ)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .encoders.ecg_encoder import build_ecg_encoder
from .encoders.localizer_encoder import build_localizer_encoder
from .encoders.tabular_encoder import build_tabular_encoder


class ProjectionHead(nn.Module):
    """MLP projection head to shared embedding space."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class TriModalCLIP(nn.Module):
    """
    Tri-modal contrastive learning model (Localizer + ECG + Tabular).
    
    Architecture:
    - Localizer: [3, 224, 224] -> 768-dim
    - ECG: [12, 5000] -> 384-dim  
    - Tabular: [num_features] -> 768-dim
    - All projected to shared projection_dim space
    
    Loss:
    - Tri-modal contrastive: L = α·L(img,txt) + β·L(txt,aud) + γ·L(aud,img)
    """
    
    def __init__(self,
                 # Encoder configs
                 ecg_model_name: str = 'vit_tiny_patchX',
                 ecg_patch_size: int = 100,
                 localizer_model_name: str = 'vit_base_localizer',
                 localizer_multi_view: bool = False,
                 localizer_aggregation: str = 'mean',
                 tabular_embed_dim: int = 384,
                 tabular_input_size: int = 233,
                 tabular_depth: int = 4,
                 tabular_num_heads: int = 8,
                 # Projection configs
                 projection_dim: int = 768,
                 hidden_dim: int = 1024,
                 projection_dropout: float = 0.1,
                 init_temperature: float = 0.07,
                 learnable_temperature: bool = True,
                 # Pretrained weights
                 ecg_pretrained_path: Optional[str] = None,
                 localizer_pretrained_path: Optional[str] = None,
                 tabular_pretrained_path: Optional[str] = None):
        super().__init__()
        
        self.projection_dim = projection_dim
        
        # ECG Encoder: [batch, 12, 5000] -> [batch, 384]
        self.ecg_encoder = build_ecg_encoder(
            model_name=ecg_model_name,
            global_pool=True,
            patch_size=ecg_patch_size,
            pretrained_path=ecg_pretrained_path
        )
        self.ecg_embed_dim = self.ecg_encoder.embed_dim
        
        # Localizer Encoder: [batch, 3, 224, 224] -> [batch, 768]
        self.localizer_encoder = build_localizer_encoder(
            model_name=localizer_model_name,
            global_pool=True,
            multi_view_mode=localizer_multi_view,
            aggregation_method=localizer_aggregation,
            pretrained_path=localizer_pretrained_path
        )
        self.localizer_embed_dim = self.localizer_encoder.embed_dim
        
        # Tabular Encoder: [batch, num_features] -> [batch, 384]
        self.tabular_encoder = build_tabular_encoder(
            embed_dim=tabular_embed_dim,
            input_size=tabular_input_size,
            depth=tabular_depth,
            num_heads=tabular_num_heads,
            include_biomarkers=False,
            pretrained_path=tabular_pretrained_path
        )
        self.tabular_embed_dim = self.tabular_encoder.embed_dim
        
        # project all encoders to shared projection_dim space
        self.ecg_projection = ProjectionHead(
            input_dim=self.ecg_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            dropout=projection_dropout
        )
        
        self.localizer_projection = ProjectionHead(
            input_dim=self.localizer_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            dropout=projection_dropout
        )
        
        self.tabular_projection = ProjectionHead(
            input_dim=self.tabular_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            dropout=projection_dropout
        )
        
        # temp implementation like Synergy-CLIP and ViTa
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.ones([]) * init_temperature)
        else:
            self.register_buffer('temperature', torch.tensor(init_temperature))
        
        self.learnable_temperature = learnable_temperature
    
    def encode_ecg(self, ecg: torch.Tensor, return_projection: bool = True) -> torch.Tensor:
        """Encode ECG signal."""
        ecg_emb = self.ecg_encoder(ecg)
        if return_projection:
            ecg_emb = self.ecg_projection(ecg_emb)
        return ecg_emb
    
    def encode_localizer(self, localizer: torch.Tensor, return_projection: bool = True) -> torch.Tensor:
        """Encode localizer images."""
        localizer_emb = self.localizer_encoder(localizer)
        if return_projection:
            localizer_emb = self.localizer_projection(localizer_emb)
        return localizer_emb
    
    def encode_tabular(self, tabular: torch.Tensor, return_projection: bool = True) -> torch.Tensor:
        """
        Encode tabular features.
        
        Args:
            tabular: [batch, num_features] tensor (already preprocessed)
            return_projection: If True, return projected embedding
        """
        tabular_emb = self.tabular_encoder(tabular)
        if return_projection:
            tabular_emb = self.tabular_projection(tabular_emb)
        return tabular_emb
    
    def forward(self, 
                localizer: torch.Tensor,
                ecg: torch.Tensor,
                tabular: torch.Tensor,
                return_embeddings: bool = False) -> Tuple:
        """
        Forward pass through all encoders and projections.
        
        Args:
            localizer: [batch, 3, 224, 224]
            ecg: [batch, 12, 5000]
            tabular: [batch, num_features] (preprocessed tensor)
            return_embeddings: If True, return embeddings; else similarities
        
        Returns:
            If return_embeddings=True: (localizer_emb, ecg_emb, tabular_emb)
            If return_embeddings=False: (similarities, temperature)
        """
        # Encode and project all modalities
        localizer_emb = self.encode_localizer(localizer, return_projection=True)
        ecg_emb = self.encode_ecg(ecg, return_projection=True)
        tabular_emb = self.encode_tabular(tabular, return_projection=True)
        
        # L2 normalize embeddings
        localizer_emb = F.normalize(localizer_emb, p=2, dim=-1)
        ecg_emb = F.normalize(ecg_emb, p=2, dim=-1)
        tabular_emb = F.normalize(tabular_emb, p=2, dim=-1)
        
        if return_embeddings:
            return localizer_emb, ecg_emb, tabular_emb
        
        # compute pairwise similarities with single temperature

        sim_localizer_ecg = torch.matmul(localizer_emb, ecg_emb.T) / self.temperature
        sim_ecg_tabular = torch.matmul(ecg_emb, tabular_emb.T) / self.temperature
        sim_tabular_localizer = torch.matmul(tabular_emb, localizer_emb.T) / self.temperature
        
        similarities = {
            'localizer_ecg': sim_localizer_ecg,
            'ecg_tabular': sim_ecg_tabular,
            'tabular_localizer': sim_tabular_localizer
        }
        
        temperature_value = self.temperature.item()
        
        return similarities, temperature_value
    
    def get_localizer_embedding(self, localizer):
        """Get normalized localizer embedding for downstream tasks."""
        features = self.localizer_encoder(localizer)
        embedding = self.localizer_projection(features)
        return F.normalize(embedding, dim=-1)

    def get_ecg_embedding(self, ecg):
        """Get normalized ECG embedding for downstream tasks."""
        features = self.ecg_encoder(ecg)
        embedding = self.ecg_projection(features)
        return F.normalize(embedding, dim=-1)

    def get_tabular_embedding(self, tabular):
        """Get normalized tabular embedding for downstream tasks."""
        features = self.tabular_encoder(tabular)
        embedding = self.tabular_projection(features)
        return F.normalize(embedding, dim=-1)
    
    def get_embeddings(self,
                       localizer: torch.Tensor,
                       ecg: torch.Tensor,
                       tabular: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get embeddings from all modalities for downstream tasks."""
        localizer_emb, ecg_emb, tabular_emb = self.forward(
            localizer, ecg, tabular, return_embeddings=True
        )
        
        return {
            'localizer': localizer_emb,
            'ecg': ecg_emb,
            'tabular': tabular_emb
        }
    
    def freeze_encoders(self):
        """Freeze all encoder parameters."""
        for param in self.ecg_encoder.parameters():
            param.requires_grad = False
        for param in self.localizer_encoder.parameters():
            param.requires_grad = False
        for param in self.tabular_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self):
        """Unfreeze all encoder parameters."""
        for param in self.ecg_encoder.parameters():
            param.requires_grad = True
        for param in self.localizer_encoder.parameters():
            param.requires_grad = True
        for param in self.tabular_encoder.parameters():
            param.requires_grad = True

