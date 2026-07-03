import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from timm.models.vision_transformer import Block


# 3D Positional Embedding Helper

def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """
    grid_size: int of the spatial grid (e.g., 14 for 224//16)
    t_size: int of the temporal grid (e.g., 6 for 12//2)
    Returns: [t_size * grid_size * grid_size, embed_dim] or [1 + ..., embed_dim]
    """
    assert embed_dim % 3 == 0, "Embed dim must be divisible by 3 for 3D pos_embed"
    
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_t = np.arange(t_size, dtype=np.float32)
    
    grid_t, grid_h, grid_w = np.meshgrid(grid_t, grid_h, grid_w, indexing='ij')
    
    pos_t = grid_t.reshape(-1)
    pos_h = grid_h.reshape(-1)
    pos_w = grid_w.reshape(-1)
    

    dim = embed_dim // 3
    omega = np.arange(dim // 2, dtype=np.float32)
    omega /= (dim / 2.)
    omega = 1. / 10000**omega
    
    out_t = np.einsum('m,d->md', pos_t, omega)
    out_h = np.einsum('m,d->md', pos_h, omega)
    out_w = np.einsum('m,d->md', pos_w, omega)
    
    emb_t = np.concatenate([np.sin(out_t), np.cos(out_t)], axis=1)
    emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1)
    emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1)
    
    pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, num_frames=12, tubelet_size=2, in_chans=9, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = img_size // patch_size
        self.t_size = num_frames // tubelet_size
        self.num_patches = (self.grid_size ** 2) * self.t_size
        
        # 3D Convolution to extract spatiotemporal "Tubelets"
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(tubelet_size, patch_size, patch_size), 
            stride=(tubelet_size, patch_size, patch_size)
        )

    def forward(self, x):
        # x shape: [Batch, Channels(9), Time(12), Height(224), Width(224)]
        x = self.proj(x)  # [B, Embed, T_size, H_size, W_size]
        x = x.flatten(2).transpose(1, 2)  # [B, Num_Patches, Embed]
        return x

# main model
class CineVideoMAE(pl.LightningModule):
    def __init__(self, 
                 img_size=224, patch_size=16, 
                 num_frames=12, tubelet_size=2, in_chans=9, 
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4.0, norm_pix_loss=False, 
                 mask_ratio=0.90, # 90% IS CRITICAL FOR VIDEO MAE
                 learning_rate=1.5e-4, weight_decay=0.05,
                 max_epochs=200, warmup_epochs=20):
        
        super().__init__()
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        
        # MAE encoder
        self.patch_embed = PatchEmbed3D(img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # MAE decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Predicts every pixel in the tubelet
        self.pred_dim = tubelet_size * patch_size**2 * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.pred_dim, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        # 3D pos. encodings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, self.patch_embed.t_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, self.patch_embed.t_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, T, H, W)
        x: (N, L, tubelet * patch**2 * C)
        """
        N, C, T, H, W = imgs.shape
        p = self.patch_size
        u = self.tubelet_size
        
        h_grid = H // p
        w_grid = W // p
        t_grid = T // u
        
        # Reshape to extract tubelets
        x = imgs.reshape(shape=(N, C, t_grid, u, h_grid, p, w_grid, p))
        x = torch.einsum('nctuhpwq->nthwupqc', x)
        x = x.reshape(shape=(N, t_grid * h_grid * w_grid, u * p**2 * C))
        return x

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  
        ids_shuffle = torch.argsort(noise, dim=1)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :] 
        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  
        x = torch.cat([x[:, :1, :], x_], dim=1)  

        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :] 
        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 
        loss = (loss * mask).sum() / mask.sum() 
        return loss

    def forward(self, imgs, mask_ratio=0.90):
        # imgs shape expected: [B, 9, 12, 224, 224]
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            imgs = batch[0] # Adjust based on dataloader
        else:
            imgs = batch
            
        loss, pred, mask = self.forward(imgs, mask_ratio=self.mask_ratio)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch
            
        loss, pred, mask = self.forward(imgs, mask_ratio=self.mask_ratio)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return [optimizer], [scheduler]

    def get_encoder_for_downstream(self):
        """Extracts the 4D Video Encoder for Stage 2"""
        encoder = nn.Module()
        encoder.patch_embed = self.patch_embed
        encoder.cls_token = self.cls_token
        encoder.pos_embed = self.pos_embed
        encoder.blocks = self.blocks
        encoder.norm = self.norm
        encoder.embed_dim = self.hparams.embed_dim
        
        def forward_features(x):
            # x: [B, 9, 12, 224, 224]
            x = encoder.patch_embed(x)
            x = x + encoder.pos_embed[:, 1:, :]
            cls_token = encoder.cls_token + encoder.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            for blk in encoder.blocks:
                x = blk(x)
            x = encoder.norm(x)
            return x[:, 0] # Return CLS token representing the whole video
        
        encoder.forward = forward_features
        return encoder