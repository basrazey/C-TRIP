import torch
import torch.nn as nn

class FactorizedVideoViT(nn.Module):
    """
    Extracts spatial features per-frame, then applies temporal attention across frames.
    """
    def __init__(self, spatial_encoder, embed_dim=768, num_frames=12, num_temporal_layers=4):
        super().__init__()
        self.spatial_encoder = spatial_encoder
        self.embed_dim = embed_dim
        
        # temporal positional embeddings (which tells the model frame 0 is ED, frame 6 is ES, etc.)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=.02)
        
        # temporal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=embed_dim * 4, 
            activation='gelu', 
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_temporal_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [Batch, Time, Channels(Slices), H, W]
        B, T, C, H, W = x.shape
        

        x_spatial = x.view(B * T, C, H, W)
        
        spatial_feats = self.spatial_encoder(x_spatial)

        x_temporal = spatial_feats.view(B, T, self.embed_dim)
        

        x_temporal = x_temporal + self.temporal_pos_embed[:, :T, :]
        
       
        x_temporal = self.temporal_transformer(x_temporal)

        video_feat = self.norm(x_temporal.mean(dim=1))
        
        return video_feat