import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
from ..utils.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid
import wandb


class CineMAE(pl.LightningModule):
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 in_chans=3, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12,
                 decoder_embed_dim=512, 
                 decoder_depth=8, 
                 decoder_num_heads=16,
                 mlp_ratio=4.0,
                 norm_pix_loss=False, 
                 mask_ratio=0.75,
                 learning_rate=1.5e-4,
                 weight_decay=0.05,
                 max_epochs=400,
                 warmup_epochs=20):
        super().__init__()
        self.save_hyperparameters()

        # Config
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        
        
        # MAE encoder 
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        
        # MAE decoder 
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

        
    def initialize_weights(self):
        # Calculate grid size (e.g. 14 for 224x224 img with 16x16 patch)
        grid_dim = int(self.patch_embed.num_patches**.5)
    
        grid_size = (grid_dim, grid_dim) 
        
 
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], grid_size, cls_token=True)
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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
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
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
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
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :] # add pos embed w/o cls token

        # masking
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
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :] # remove cls token
        return x

    def forward_loss(self, imgs, pred, mask):
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
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def training_step(self, batch, batch_idx):
        # batch is just images 

        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
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
        
        # Visualize reconstruction for the first batch of epoch
        if batch_idx == 0:
            self._log_visualization(imgs, pred, mask)
        return loss

    def _log_visualization(self, imgs, pred, mask):
        """Log 1st sample reconstruction to WandB"""
        if not (hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log')): return
        
        with torch.no_grad():
            img = imgs[0:1] # [1, 3, H, W]
            mask_s = mask[0:1]
            pred_s = pred[0:1]
            
            # Reconstruction
            rec = self.unpatchify(pred_s) # [1, 3, H, W]
            
            # Masked View
            patches = self.patchify(img)
            mask_exp = mask_s.unsqueeze(-1).repeat(1, 1, patches.shape[-1])
            patches[mask_exp==1] = 0
            masked_img = self.unpatchify(patches)

            # Log middle slice (idx 1)
            slice_idx = self.in_chans // 2 
            
            self.logger.experiment.log({
                "val/orig": wandb.Image(img[0, slice_idx].cpu().numpy()),
                "val/masked": wandb.Image(masked_img[0, slice_idx].cpu().numpy()),
                "val/recon": wandb.Image(rec[0, slice_idx].cpu().numpy()),
            })

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return [optimizer], [scheduler]

    def get_encoder_for_downstream(self):
        """Extracts the encoder for Stage 2 (Contrastive)"""
        encoder = nn.Module()
        encoder.patch_embed = self.patch_embed
        encoder.cls_token = self.cls_token
        encoder.pos_embed = self.pos_embed
        encoder.blocks = self.blocks
        encoder.norm = self.norm
        encoder.embed_dim = self.embed_dim
        
        def forward_features(x):
            x = encoder.patch_embed(x)
            x = x + encoder.pos_embed[:, 1:, :]
            cls_token = encoder.cls_token + encoder.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            for blk in encoder.blocks:
                x = blk(x)
            x = encoder.norm(x)
            return x[:, 0] # Return CLS token
        
        encoder.forward = forward_features
        return encoder