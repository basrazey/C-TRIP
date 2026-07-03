"""
Masked Autoencoder for Localizer MRI Images.
Pretrain on localizer MRI volumes treating slices as channels.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from functools import partial
import numpy as np

from timm.models.vision_transformer import PatchEmbed, Block
from src.utils.pos_embed import get_2d_sincos_pos_embed


class LocalizerMRI_MAE(pl.LightningModule):
    """
    Masked Autoencoder for MRI Localizer images.
    Treats k slices as input channels [B, k, H, W].
    """
    
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # MRI parameters
        self.img_size = hparams.img_size if hasattr(hparams, 'img_size') else 224  # 224
        self.patch_size = hparams.patch_size if hasattr(hparams, 'patch_size') else 16  # 16
        self.in_chans = hparams.slice_k if hasattr(hparams, 'slice_k') else 3  # Number of slices as channels (e.g., 3)
        self.embed_dim = hparams.embed_dim if hasattr(hparams, 'embed_dim') else 768  # 768
        self.depth = hparams.depth if hasattr(hparams, 'depth') else 12  # 12
        self.num_heads = hparams.num_heads if hasattr(hparams, 'num_heads') else 12  # 12
        self.decoder_embed_dim = hparams.decoder_embed_dim if hasattr(hparams, 'decoder_embed_dim') else 512  # 512
        self.decoder_depth = hparams.decoder_depth if hasattr(hparams, 'decoder_depth') else 8  # 8
        self.decoder_num_heads = hparams.decoder_num_heads if hasattr(hparams, 'decoder_num_heads') else 16  # 16
        self.mlp_ratio = hparams.mlp_ratio if hasattr(hparams, 'mlp_ratio') else 4.0  # 4.0
        self.norm_pix_loss = hparams.norm_pix_loss if hasattr(hparams, 'norm_pix_loss') else False  # False
        self.mask_ratio = hparams.mask_ratio if hasattr(hparams, 'mask_ratio') else 0.75  # 0.75
        
        # Training parameters
        self.learning_rate = hparams.learning_rate if hasattr(hparams, 'learning_rate') else 1e-3
        self.weight_decay = hparams.weight_decay if hasattr(hparams, 'weight_decay') else 0.05
        self.warmup_epochs = hparams.warmup_epochs if hasattr(hparams, 'warmup_epochs') else 10
        self.max_epochs = hparams.max_epochs if hasattr(hparams, 'max_epochs') else 100
        
 
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, 
            patch_size=self.patch_size, 
            in_chans=self.in_chans,  # k slices as channels
            embed_dim=self.embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(self.depth)])
        self.norm = nn.LayerNorm(self.embed_dim)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(self.decoder_depth)])

        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_size**2 * self.in_chans, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        grid_dim = int(self.patch_embed.num_patches**.5)
        grid_size = (grid_dim, grid_dim)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W) where C = slice_k
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*C]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def training_step(self, batch, batch_idx):
        # Batch format: (ecg, mri, pid, targets) - we only use mri
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            _, mri, _, _ = batch
        else:
            mri = batch
        
        loss, pred, mask = self.forward(mri, mask_ratio=self.mask_ratio)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log additional metrics
        mask_ratio_actual = mask.float().mean()
        self.log('train_mask_ratio', mask_ratio_actual, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            _, mri, _, _ = batch
        else:
            mri = batch

        loss, pred, mask = self.forward(mri, mask_ratio=self.mask_ratio)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Log reconstruction quality metrics
        if batch_idx == 0 or batch_idx % 10 == 0: # Log reconstruction examples in every ten epochs
            self._log_reconstruction_examples(mri, pred, mask)
        
        return loss

    def _log_reconstruction_examples(self, imgs, pred, mask):
        """Log reconstruction examples to visualize MAE quality."""
        with torch.no_grad():
            # Take first image from batch
            img_orig = imgs[0:1]  # [1, C, H, W]
            pred_patches = pred[0:1]  # [1, L, p*p*C]
            mask_patches = mask[0:1]  # [1, L]
            
            # Reconstruct full image
            img_pred = self.unpatchify(pred_patches)  # [1, C, H, W]
            
            # Create masked version (set masked patches to 0)
            patches_orig = self.patchify(img_orig)  # [1, L, p*p*C]
            patches_masked = patches_orig.clone()
            patches_masked[mask_patches.unsqueeze(-1).repeat(1, 1, patches_orig.shape[-1]) == 1] = 0
            img_masked = self.unpatchify(patches_masked)
            
            # Log to wandb (if available)
            try:
                import wandb
                if wandb.run is not None:
                    # Convert to numpy for logging (take middle slice for visualization)
                    mid_slice = self.in_chans // 2
                    
                    wandb.log({
                        "reconstruction/original": wandb.Image(img_orig[0, mid_slice].cpu().numpy()),
                        "reconstruction/masked": wandb.Image(img_masked[0, mid_slice].cpu().numpy()),
                        "reconstruction/predicted": wandb.Image(img_pred[0, mid_slice].cpu().numpy()),
                        "epoch": self.current_epoch
                    })
            except:
                pass

    def configure_optimizers(self):
        # AdamW optimizer with cosine annealing
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

    def get_encoder_for_downstream(self):
        """
        Extract encoder for downstream tasks.
        Returns the encoder without decoder for feature extraction.
        """
        encoder = nn.Module()
        encoder.patch_embed = self.patch_embed
        encoder.cls_token = self.cls_token
        encoder.pos_embed = self.pos_embed
        encoder.blocks = self.blocks
        encoder.norm = self.norm
        encoder.embed_dim = self.embed_dim
        
        def forward_features(x):
            # embed patches
            x = encoder.patch_embed(x)
            # add pos embed w/o cls token
            x = x + encoder.pos_embed[:, 1:, :]
            
            # append cls token
            cls_token = encoder.cls_token + encoder.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # apply Transformer blocks
            for blk in encoder.blocks:
                x = blk(x)
            x = encoder.norm(x)
            
            return x[:, 0]  # return cls token
        
        encoder.forward = forward_features
        return encoder


# Factory functions for different model sizes
def localizer_mae_tiny(**kwargs):
    model = LocalizerMRI_MAE(
        embed_dim=384, depth=6, num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, **kwargs)
    return model

def localizer_mae_small(**kwargs):
    model = LocalizerMRI_MAE(
        embed_dim=512, depth=8, num_heads=8,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, **kwargs)
    return model

def localizer_mae_base(**kwargs):
    model = LocalizerMRI_MAE(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, **kwargs)
    return model

def localizer_mae_large(**kwargs):
    model = LocalizerMRI_MAE(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, **kwargs)
    return model


