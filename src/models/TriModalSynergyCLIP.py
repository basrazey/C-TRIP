from calendar import c
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torchmetrics import R2Score
from src.networks.LocalizerEncoder import LocalizerMRI_MAE, localizer_mae_base
from src.networks.ECGEncoder import vit_tiny_patchX
from src.networks.TabularEncoder import TabularViT
from src.networks.videoViT import FactorizedVideoViT
from dataclasses import asdict
from argparse import Namespace
from sklearn.manifold import TSNE  # Standard library, no Numba dependency
try:
    import umap
    UMAP_AVAILABLE = True
except (ImportError, RuntimeError, OSError) as e:
    print(f" UMAP/Numba initialization failed: {e}")
    print("UMAP visualization will be disabled for this run.")
    UMAP_AVAILABLE = False
import matplotlib.pyplot as plt
import wandb
from torchmetrics import Accuracy
from torchvision.models import resnet50
import torch.nn as nn
import os

class TriModalSynergyCLIP(pl.LightningModule):
    def __init__(self, config, vocab_sizes,target_mean, target_std, mri_chkpt_path=None,
                 ecg_chkpt_path=None, tabular_chkpt_path=None,
                 cine_chkpt_path=None):
        super().__init__()
        self.save_hyperparameters(ignore=['mri_chkpt_path', 'ecg_chkpt_path', 'tabular_chkpt_path', 'target_mean', 'target_std'])
        self.config = config
        self.register_buffer('target_mean', target_mean)
    
        self.register_buffer('target_std', target_std)
        
        self.use_mri = getattr(config, 'use_mri', True)
        self.use_localizer_12s = False # getattr(config, 'use_localizer_12s', False)
        self.use_ecg = getattr(config, 'use_ecg', True)
        self.use_tabular = getattr(config, 'use_tabular', True)
        self.use_cine = getattr(config, 'use_cine', False) or getattr(config, 'use_cine_12_frames', False) or getattr(config, 'use_cine_24_frames', False)
        
        #self.temperature = getattr(config, 'temperature', 0.07)
        self.temp_mri_ecg = getattr(config, 'temp_mri_ecg', 0.1)
        self.temp_mri_tab = getattr(config, 'temp_mri_tab', 0.25)
        self.use_ecg_tab_loss = getattr(config, 'use_ecg_tab_loss', False)
        self.temp_ecg_tab = getattr(config, 'temp_ecg_tab', 0.1)
        self.temp_cine_mri = getattr(config, 'temp_cine_mri', 0.1)
        self.temp_cine_tab = getattr(config, 'temp_cine_tab', 0.1)
        self.temp_cine_ecg = getattr(config, 'temp_cine_ecg', 0.1)
        
        
        #lambda weights for losses - can be tuned to balance the different contrastive objectives
        self.w_mri_ecg = getattr(config, 'w_mri_ecg', 1.0)
        self.w_mri_tab = getattr(config, 'w_mri_tab', 1.0)
        self.w_ecg_tab = getattr(config, 'w_ecg_tab', 1.0)
        
        self.w_cine_mri = getattr(config, 'w_cine_mri', 1.0)
        self.w_cine_tab = getattr(config, 'w_cine_tab', 1.0)
        self.w_cine_ecg = getattr(config, 'w_cine_ecg', 1.0)
        
        self.weight_decay = getattr(config, 'weight_decay', 1e-4)
        
        self.fn_threshold = getattr(config, 'fn_threshold', 0.95) # Threshold for false negatives
        
        mri_embed_dim = 0
        ecg_embed_dim = 0
        tab_embed_dim = 0
        cine_embed_dim = 0
       
        
        # Encoders
        
        #  MRI Encoder Pre-trained MAE 
        if self.use_mri:
            mri_in_chans = 12 if self.use_localizer_12s else 3
            print(f"Initializing MRI Encoder with {mri_in_chans} input channels (localizer_12s={self.use_localizer_12s})")
            
        
            
            if mri_chkpt_path:
                print(f"Loading MRI MAE from {mri_chkpt_path}")
                
                mri_args_dict = asdict(config).copy()
                
                # force the standard ViT-Base parameters 
                mri_args_dict['embed_dim'] = 768
                mri_args_dict['num_heads'] = 12
                mri_args_dict['depth'] = 12
                mri_args_dict['mlp_ratio'] = 4.0
                
                mri_args_dict['in_chans'] = mri_in_chans  
                mri_args_dict['slice_k'] = mri_in_chans
                
                # make decoder params exist (even though discarded later)
                # just to prevent init crashes
                mri_args_dict['decoder_embed_dim'] = 512 
                mri_args_dict['decoder_depth'] = 8
                mri_args_dict['decoder_num_heads'] = 16
                
                
                hparams_ns = Namespace(**mri_args_dict)
                mae_model = LocalizerMRI_MAE(hparams_ns)
                checkpoint = torch.load(mri_chkpt_path, map_location='cpu', weights_only=False)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
    
                keys = mae_model.load_state_dict(state_dict, strict=False)
                print(f"MRI Weights Loaded. Missing keys: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)}")
                self.mri_encoder = mae_model.get_encoder_for_downstream()
                mri_embed_dim = mae_model.embed_dim

            else:
                # define standard ViT-Base parameters manually
                # replicate the config structure expected by LocalizerMRI_MAE
                print(f"Initializing MRI Encoder from SCRATCH with {mri_in_chans} channels.")
                random_init_args = asdict(config).copy()
                random_init_args.update({
                    'img_size': 224,
                    'in_chans': mri_in_chans,
                    'slice_k': mri_in_chans,
                    'embed_dim': 768,
                    'depth': 12,
                    'num_heads': 12,
                    'mlp_ratio': 4.0,
                    # Decoder args are required for init, even if unused downstream
                    'decoder_embed_dim': 512,
                    'decoder_depth': 8, 
                    'decoder_num_heads': 16,
                    'mask_ratio': 0.75,
                    'norm_pix_loss': False
                })
                
            
                hparams_ns = Namespace(**random_init_args)
                
                
                # initialize
                mae_model = LocalizerMRI_MAE(hparams_ns)
                self.mri_encoder = mae_model.get_encoder_for_downstream()
                mri_embed_dim  = 768                
                
                
        else:
            print("MRI not used.")
        
        if self.use_ecg:
            #  ECG Encoder (1D ViT) 
            self.ecg_encoder = vit_tiny_patchX(
                img_size=(12, 5000), 
                patch_size=(1, 100), 
                in_chans=1, 
                num_classes=0 
            )
            if ecg_chkpt_path:
                print(f"Loading ECG ViT from {ecg_chkpt_path}")
                checkpoint = torch.load(ecg_chkpt_path, map_location='cpu', weights_only=False)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                    
                state_dict = {k.replace('module.', '').replace('model.', ''): v for k, v in state_dict.items()}

                keys = self.ecg_encoder.load_state_dict(state_dict, strict=False)
                print(f"ECG Weights Loaded. Missing keys: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)}")
            else:
                print("WARNING: Initializing Random ECG Encoder")
            ecg_embed_dim = self.ecg_encoder.embed_dim if hasattr(self.ecg_encoder, 'embed_dim') else 384
       
       
       
        if self.use_tabular:
            #  Tabular Encoder (ViT) 
            #  put ALL categoricals into 'multi_categorical' because sizes > 2
            multi_cat_dict = {}
            for name, size in vocab_sizes.items():
                # Tuple format: (vocab_size, embedding_dim)
                multi_cat_dict[name] = (size, config.embed_dim) 

            feature_names_struct = {
                "numerical": config.numerical_features,
                "single_categorical": [], # Leave empty to avoid crash
                "multi_categorical": multi_cat_dict
            }
            
            # the args object required by TabularViT
            class TabArgs:
                grad_checkpointing = False
                embedding_dim = config.tab_embed_dim # e.g. 768
                encoder_num_heads = 8
                encoder_mlp_ratio = 4
                encoder_num_layers = 6
            
            self.tab_encoder = TabularViT(TabArgs(), feature_names_struct)
            tab_embed_dim = TabArgs.embedding_dim
        
        
            if tabular_chkpt_path:
                import sys
                import __main__
                
                # create a dummy class
                class DummyConfig:
                    pass
                setattr(__main__, "GlobalConfig", DummyConfig)
                setattr(__main__, "ModelConfig", DummyConfig)
                setattr(__main__, "DataConfig", DummyConfig)
                setattr(__main__, "TrainConfig", DummyConfig)
                print(f"Loading Tabular ViT from {tabular_chkpt_path}")
                checkpoint = torch.load(tabular_chkpt_path, map_location='cpu', weights_only=False)
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                    
                encoder_dict = {k: v for k, v in state_dict.items() if 'decoder' not in k and 'mask_token' not in k}
                
                keys = self.tab_encoder.load_state_dict(encoder_dict, strict=False)
                print(f"Tabular MAE Weights Loaded. Missing: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)}")
                
            else:
                print("Initializing Random Tabular Encoder")
        else:
            print("Tabular not used.")
            
            
            
            
            
        if self.use_cine:
            print("Initializing cine encoder.")
            
            # 4D video MAE (12 frames, 9 slices)
            if getattr(config, 'use_cine_12f_9s', False): 
                print("Initializing 4D VideoMAE (9 channels, 12 frames).")
                
                # Import new 4D class
                from src.models.CineVideoMAE import CineVideoMAE 
                
                #cine_args = asdict(config).copy()
                cine_args = {
                    'img_size': 224, 
                    'patch_size': 16,
                    'num_frames': 12, 
                    'tubelet_size': 2, 
                    'in_chans': 9,
                    'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4.0,
                    'decoder_embed_dim': 384, 'decoder_depth': 4, 'decoder_num_heads': 8,
                    'mask_ratio': 0.90
                }
                
                # initialize the 4D model
                video_mae_model = CineVideoMAE(**cine_args)
                print(f"Initialized 4D VideoMAE with {cine_args['in_chans']} channels and {cine_args['num_frames']} frames.")
                
                self.cine_encoder = video_mae_model.get_encoder_for_downstream()
                cine_embed_dim = 768
                
                if cine_chkpt_path:
                    print(f"Loading 4D Cine VideoMAE Weights from {cine_chkpt_path}")
                    ckpt = torch.load(cine_chkpt_path, map_location='cpu', weights_only=False)
                    state_dict = ckpt.get('encoder_state_dict', ckpt.get('state_dict', ckpt))
                    
                    keys = self.cine_encoder.load_state_dict(state_dict, strict=False)
                    print(f"4D Cine ViT Loaded. Missing: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)}")

            # 
            # factorized 2D+1D (1, 12, or 24 frames, 3 slices)
            # 
            else:
                cine_in_chans = 3
                num_frames = 1
                if getattr(config, 'use_cine_12_frames', False): 
                    cine_in_chans = 36  # 12 frames * 3 slices
                    print(f"Initializing Cine Encoder for 12-Frame Channel Stacking ({cine_in_chans} channels).")
                elif getattr(config, 'use_cine_24_frames', False): 
                    cine_in_chans = 72  # 24 frames * 3 slices
                    print(f"Initializing Cine Encoder for 24-Frame Channel Stacking ({cine_in_chans} channels).")

                else:
                    print(f"Initializing standard 3-slice cine encoder with {cine_in_chans} input channels.")

                cine_args = asdict(config).copy()
                cine_args.update({
                    'img_size': 224, 
                    'in_chans': cine_in_chans, 
                    'slice_k': cine_in_chans,
                    'embed_dim': 768, 'num_heads': 12, 'depth': 12, 'mlp_ratio': 4.0,
                    'decoder_embed_dim': 512, 'decoder_depth': 8
                })
                
                hparams_ns = Namespace(**cine_args)
                spatial_mae_model = LocalizerMRI_MAE(hparams_ns)
                self.spatial_cine_encoder = spatial_mae_model.get_encoder_for_downstream()
                self.cine_encoder = spatial_mae_model.get_encoder_for_downstream()
                cine_embed_dim = 768
                
                
                
                if cine_chkpt_path:
                    print(f"Loading Factorized Cine ViT Weights from {cine_chkpt_path}")
                    ckpt = torch.load(cine_chkpt_path, map_location='cpu', weights_only=False)
                    state_dict = ckpt.get('encoder_state_dict', ckpt.get('state_dict', ckpt))
                    
                    keys = self.spatial_cine_encoder.load_state_dict(state_dict, strict=False)
                    print(f"Factorized Cine ViT Loaded. Missing: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)}")
                
                
        # rojection heads (latent alignment)

        self.proj_dim = config.projection_dim
        
        def build_head(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, out_dim)
            )

        if self.use_mri: self.mri_proj = build_head(mri_embed_dim, self.proj_dim)
        if self.use_ecg: self.ecg_proj = build_head(ecg_embed_dim, self.proj_dim)
        if self.use_tabular: self.tab_proj = build_head(tab_embed_dim, self.proj_dim)
        if self.use_cine: self.cine_proj = build_head(cine_embed_dim, self.proj_dim)

        # temperatures for contrastive losses - initialized as learnable parameters                   
            
        if self.use_mri: self.logit_scale_cine_mri = nn.Parameter(torch.ones([]) * np.log(1 / config.temp_cine_mri))
        if self.use_tabular: self.logit_scale_cine_tab = nn.Parameter(torch.ones([]) * np.log(1 / config.temp_cine_tab))
        if self.use_ecg: self.logit_scale_cine_ecg = nn.Parameter(torch.ones([]) * np.log(1 / config.temp_cine_ecg))
                    

        if self.use_mri and self.use_ecg: self.logit_scale_mri_ecg = nn.Parameter(torch.ones([]) * np.log(1 / self.temp_mri_ecg))
        if self.use_mri and self.use_tabular: self.logit_scale_mri_tab = nn.Parameter(torch.ones([]) * np.log(1 / self.temp_mri_tab))
        if self.use_ecg and self.use_tabular and self.use_ecg_tab_loss: self.logit_scale_ecg_tab = nn.Parameter(torch.ones([]) * np.log(1 / self.temp_ecg_tab))
        
        
        
        # initialize these dynamically in setup() or lazily to handle batch_size changes
        self.train_acc_top1 = Accuracy(task="multiclass", num_classes=config.batch_size, top_k=1)
        self.train_acc_top5 = Accuracy(task="multiclass", num_classes=config.batch_size, top_k=5)
        self.val_acc_top1 = Accuracy(task="multiclass", num_classes=config.batch_size, top_k=1)
        self.val_acc_top5 = Accuracy(task="multiclass", num_classes=config.batch_size, top_k=5)
        
        
        self.val_preds_storage = []


    def forward(self, mri, ecg, tabular, cine):
        mri_feat, ecg_feat, tab_feat, cine_feat = None, None, None, None

        if self.use_mri and mri is not None:
            mri_feat = self.mri_encoder(mri)
            
        if self.use_ecg and ecg is not None:
            ecg_feat = self.ecg_encoder.forward_features(ecg)
            
        if self.use_tabular and tabular is not None:
            tab_seq = self.tab_encoder(tabular)
            tab_feat = tab_seq[:, 0, :]
        
        if self.use_cine and cine is not None:
            # ResNet output: [B, 2048, 1, 1]
            cine_feat = self.cine_encoder(cine)
            
        return mri_feat, ecg_feat, tab_feat, cine_feat
    
    
    
    def get_local_features(self, mri=None, ecg=None):
        """ for Dense/Local alignment visualization"""
        mri_local, ecg_local = None, None
        
        # MRI: Forward pass but skip the pooling/CLS selection if possible
        if self.use_mri and mri is not None:
             #  encoder has forward_features that returns [B, N, D]
             if hasattr(self.mri_encoder, 'forward_features'):
                 mri_local = self.mri_encoder.forward_features(mri)
                 # Remove CLS if present
                 if mri_local.shape[1] > 196: # e.g. 197
                     mri_local = mri_local[:, 1:, :]
        
        # ECG: Get local patch tokens
        if self.use_ecg and ecg is not None:
            ecg_local = self.ecg_encoder.forward_features(ecg, localized=True) # Assuming adapted encoder
            # If standard ViT, just call forward features
            if ecg_local is None: 
                ecg_local = self.ecg_encoder.forward_features(ecg)

        return mri_local, ecg_local
    
    
    def compute_robust_clip_loss(self, emb1, emb2, logit_scale_param, log_prefix, train=True):
        """
        Robust CLIP Loss that handles False Negatives via Thresholding
        AND calculates Top-K Accuracy.
        """
        logit_scale = logit_scale_param.exp()
        logits = logit_scale * (emb1 @ emb2.t())
        batch_size = logits.shape[0]
        
        if batch_size < 2:
            # Cannot compute contrastive loss/accuracy for a single sample (no negatives)
            # Return 0.0 loss to avoid crash
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        labels = torch.arange(batch_size, device=logits.device)
                # Update metric objects
        if train:
            # Re-init if batch size changed (last batch may be smaller)
            if self.train_acc_top1.num_classes != batch_size:
                self.train_acc_top1 = Accuracy(task="multiclass", num_classes=batch_size, top_k=1).to(logits.device)
                self.train_acc_top5 = Accuracy(task="multiclass", num_classes=batch_size, top_k=min(5, batch_size)).to(logits.device)
            
            acc1 = self.train_acc_top1(logits, labels)
            acc5 = self.train_acc_top5(logits, labels)
            self.log(f"train/{log_prefix}_top1", acc1)
            self.log(f"train/{log_prefix}_top5", acc5)
        else:
             if self.val_acc_top1.num_classes != batch_size:
                self.val_acc_top1 = Accuracy(task="multiclass", num_classes=batch_size, top_k=1).to(logits.device)
                self.val_acc_top5 = Accuracy(task="multiclass", num_classes=batch_size, top_k=min(5, batch_size)).to(logits.device)
             
             acc1 = self.val_acc_top1(logits, labels)
             acc5 = self.val_acc_top5(logits, labels)
             self.log(f"val/{log_prefix}_top1", acc1, on_epoch=True)

        #
        # If a negative pair has high cosine sim, ignore it in the loss Standard CrossEntropy includes all negatives in the denominator.
        # stick to standard CE unless it has explicit FN labels and visualize the FN rate:
        
        with torch.no_grad():
            # Check how many off-diagonal elements are > threshold
            sim_matrix = (emb1 @ emb2.t())
            mask = torch.eye(batch_size, device=sim_matrix.device).bool()
            off_diag = sim_matrix[~mask]
            false_negatives = (off_diag > self.fn_threshold).float().mean()
            self.log(f"metrics/{log_prefix}_fn_rate", false_negatives)

        # Standard Symmetric Loss
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        total_loss = (loss_a + loss_b) / 2
        if train:
            self.log(f"train/loss_{log_prefix}", total_loss)
        else:
            self.log(f"val/loss_{log_prefix}", total_loss, on_epoch=True)
        return total_loss
    

    def training_step(self, batch, batch_idx):
        # Batch unpacking
        mri = batch.get('mri')
        ecg = batch.get('ecg')
        tabular = batch.get('tabular')
        cine = batch.get('cine')
        
        
        #  don't necessarily need labels for contrastive, 
        # Get representations
        mri_feat, ecg_feat, tab_feat, cine_feat = self(mri, ecg, tabular, cine)
        
        #project & normalize
        mri_emb = F.normalize(self.mri_proj(mri_feat), dim=-1) if mri_feat is not None else None
        ecg_emb = F.normalize(self.ecg_proj(ecg_feat), dim=-1) if ecg_feat is not None else None
        tab_emb = F.normalize(self.tab_proj(tab_feat), dim=-1) if tab_feat is not None else None
        cine_emb = F.normalize(self.cine_proj(cine_feat), dim=-1) if cine_feat is not None else None
        
        
        weighted_losses = []
        total_weight = 0.0
        log_dict = {}
        
        #  T = 1 / exp(logit_scale)
        if hasattr(self, 'logit_scale_mri_ecg'):
            current_temp_mri_ecg = 1.0 / self.logit_scale_mri_ecg.exp()
            self.log("train/temp_mri_ecg", current_temp_mri_ecg)

        if hasattr(self, 'logit_scale_mri_tab'):
            current_temp_mri_tab = 1.0 / self.logit_scale_mri_tab.exp()
            self.log("train/temp_mri_tab", current_temp_mri_tab)
        if hasattr(self, 'logit_scale_ecg_tab'):
            current_temp_ecg_tab = 1.0 / self.logit_scale_ecg_tab.exp()
            self.log("train/temp_ecg_tab", current_temp_ecg_tab)
        if hasattr(self, 'logit_scale_cine_mri'):
            current_temp_cine_mri = 1.0 / self.logit_scale_cine_mri.exp()
            self.log("train/temp_cine_mri", current_temp_cine_mri)
        if hasattr(self, 'logit_scale_cine_tab'):
            current_temp_cine_tab = 1.0 / self.logit_scale_cine_tab.exp()
            self.log("train/temp_cine_tab", current_temp_cine_tab)
        # Compute pairwise losses
            
        
        if mri_emb is not None and ecg_emb is not None:
            loss_mri_ecg = self.compute_robust_clip_loss(mri_emb, ecg_emb, self.logit_scale_mri_ecg, "mri_ecg", train=True)
            weighted_losses.append(self.w_mri_ecg * loss_mri_ecg)
            total_weight += self.w_mri_ecg
            log_dict["loss/mri_ecg"] = loss_mri_ecg
        if mri_emb is not None and tab_emb is not None:
            loss_mri_tab = self.compute_robust_clip_loss(mri_emb, tab_emb, self.logit_scale_mri_tab, "mri_tab", train=True)
            weighted_losses.append(self.w_mri_tab * loss_mri_tab)
            total_weight += self.w_mri_tab
            log_dict["loss/mri_tab"] = loss_mri_tab
        if ecg_emb is not None and tab_emb is not None and self.use_ecg_tab_loss:
            loss_ecg_tab = self.compute_robust_clip_loss(ecg_emb, tab_emb, self.logit_scale_ecg_tab, "ecg_tab", train=True)
            weighted_losses.append(self.w_ecg_tab * loss_ecg_tab)
            total_weight += self.w_ecg_tab
            log_dict["loss/ecg_tab"] = loss_ecg_tab
            
        if cine_emb is not None and mri_emb is not None:
            loss_cine_mri = self.compute_robust_clip_loss(cine_emb, mri_emb, self.logit_scale_cine_mri, "cine_mri", train=True)
            weighted_losses.append(self.w_cine_mri * loss_cine_mri)
            total_weight += self.w_cine_mri
            log_dict["loss/cine_mri"] = loss_cine_mri
        
        if cine_emb is not None and tab_emb is not None:
            loss_cine_tab = self.compute_robust_clip_loss(cine_emb, tab_emb, self.logit_scale_cine_tab, "cine_tab", train=True)
            weighted_losses.append(self.w_cine_tab * loss_cine_tab)
            total_weight += self.w_cine_tab
            log_dict["loss/cine_tab"] = loss_cine_tab
        
        if cine_emb is not None and ecg_emb is not None:
            loss_cine_ecg = self.compute_robust_clip_loss(cine_emb, ecg_emb, self.logit_scale_cine_ecg, "cine_ecg", train=True)
            weighted_losses.append(self.w_cine_ecg * loss_cine_ecg)
            total_weight += self.w_cine_ecg
            log_dict["loss/cine_ecg"] = loss_cine_ecg

            
            
        if not weighted_losses:
            raise RuntimeError("No modalities available for computing loss. Check config.use_")
    
        # Average Loss
        total_loss = sum(weighted_losses) / total_weight
        log_dict["train_loss"] = total_loss
        self.log_dict(log_dict, prog_bar=True)
        return total_loss
        

    def validation_step(self, batch, batch_idx):
        mri = batch.get('mri')
        ecg = batch.get('ecg')
        tabular = batch.get('tabular')
        cine = batch.get('cine')
        labels = batch.get('label')
        # get Features (Raw)
        mri_feat, ecg_feat, tab_feat, cine_feat = self(mri, ecg, tabular, cine)
        
        # get Projections (embeddings for contrastive loss)
        mri_emb = F.normalize(self.mri_proj(mri_feat), dim=-1) if mri_feat is not None else None
        ecg_emb = F.normalize(self.ecg_proj(ecg_feat), dim=-1) if ecg_feat is not None else None
        tab_emb = F.normalize(self.tab_proj(tab_feat), dim=-1) if tab_feat is not None else None
        cine_emb = F.normalize(self.cine_proj(cine_feat), dim=-1) if cine_feat is not None else None
        
        weighted_losses = []
        total_weight = 0.0
        log_dict = {}
        if mri_emb is not None and ecg_emb is not None:
            loss_mri_ecg = self.compute_robust_clip_loss(mri_emb, ecg_emb, self.logit_scale_mri_ecg, "mri_ecg", train=False)
            weighted_losses.append(self.w_mri_ecg * loss_mri_ecg)
            total_weight += self.w_mri_ecg
            log_dict["val/loss_mri_ecg"] = loss_mri_ecg
        if mri_emb is not None and tab_emb is not None:
            loss_mri_tab = self.compute_robust_clip_loss(mri_emb, tab_emb, self.logit_scale_mri_tab, "mri_tab", train=False)
            weighted_losses.append(self.w_mri_tab * loss_mri_tab)
            total_weight += self.w_mri_tab
            log_dict["val/loss_mri_tab"] = loss_mri_tab
       
        if ecg_emb is not None and tab_emb is not None and self.use_ecg_tab_loss:
            loss_ecg_tab = self.compute_robust_clip_loss(ecg_emb, tab_emb, self.logit_scale_ecg_tab, "ecg_tab", train=False)
            weighted_losses.append(self.w_ecg_tab * loss_ecg_tab)
            total_weight += self.w_ecg_tab
            log_dict["val/loss_ecg_tab"] = loss_ecg_tab   
        
        if cine_emb is not None and mri_emb is not None:
            loss_cine_mri = self.compute_robust_clip_loss(cine_emb, mri_emb, self.logit_scale_cine_mri, "cine_mri", train=False)
            weighted_losses.append(self.w_cine_mri * loss_cine_mri)
            total_weight += self.w_cine_mri
            log_dict["val/loss_cine_mri"] = loss_cine_mri
        
        if cine_emb is not None and tab_emb is not None:
            loss_cine_tab = self.compute_robust_clip_loss(cine_emb, tab_emb, self.logit_scale_cine_tab, "cine_tab", train=False)
            weighted_losses.append(self.w_cine_tab * loss_cine_tab)
            total_weight += self.w_cine_tab
            log_dict["val/loss_cine_tab"] = loss_cine_tab   
        
        if cine_emb is not None and ecg_emb is not None:
            loss_cine_ecg = self.compute_robust_clip_loss(cine_emb, ecg_emb, self.logit_scale_cine_ecg, "cine_ecg", train=False)
            weighted_losses.append(self.w_cine_ecg * loss_cine_ecg)
            total_weight += self.w_cine_ecg
            log_dict["val/loss_cine_ecg"] = loss_cine_ecg

        total_val_loss = sum(weighted_losses) / total_weight if total_weight > 0 else 0.0
        self.log("val/loss_total", total_val_loss, prog_bar=True, sync_dist=True)
        
        storage_item = {}
        if mri_emb is not None: storage_item['mri'] = mri_emb.detach().cpu()
        if ecg_emb is not None: storage_item['ecg'] = ecg_emb.detach().cpu()
        if tab_emb is not None: storage_item['tabular'] = tab_emb.detach().cpu()
        if cine_emb is not None: storage_item['cine'] = cine_emb.detach().cpu()
        if labels is not None: storage_item['labels'] = labels.detach().cpu() #
        self.val_preds_storage.append(storage_item)
        return total_val_loss
        
    

    def compute_clip_loss(self, emb1, emb2, logit_scale_param):
        """Standard Symmetric Contrastive Loss"""
        logit_scale = logit_scale_param.exp()
        
        # cosine similarity matrix
        logits = logit_scale * (emb1 @ emb2.t())
        
        # Labels are the diagonal (0, 1, 2...) because positive pairs are at the same index
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        
        return (loss_a + loss_b) / 2
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
    
        
    #call umap visualizations after every validation epoch
    def on_validation_epoch_end(self):
        if not self.val_preds_storage:
            return
        #embedding stats to see mode collapse exist
        keys = list(self.val_preds_storage[0].keys()) # e.g., ['mri', 'ecg']
        for key in keys:
            emb_concat = torch.cat([x[key] for x in self.val_preds_storage], dim=0)
            # std dev: If this is ~0, model has collapsed
            std_dev = torch.std(emb_concat, dim=0).mean().item()
            self.log(f"val/emb_std_{key}", std_dev)
            
            # Norm should be ~1.0 if normalized
            norm = torch.norm(emb_concat, dim=1).mean().item()
            self.log(f"val/emb_norm_{key}", norm)
            
       
        
        # call the function to log umap plots
        self.log_umap_plots_to_wandb(trainer=self.trainer)
        self.log_umap_plots_to_wandb_coloured(trainer=self.trainer)
        
    def log_umap_plots_to_wandb(self, trainer):
        if not UMAP_AVAILABLE:
            return
        if not self.val_preds_storage:
            return
        should_plot = (self.current_epoch == 0) or (self.current_epoch % 10 == 0)
        if not should_plot:
            self.val_preds_storage.clear()
            return
        keys = list(self.val_preds_storage[0].keys()) # e.g., ['mri', 'ecg']
        
        embeddings_map = {}
        for key in keys:
            embeddings_map[key] = torch.cat([x[key] for x in self.val_preds_storage], dim=0).cpu().numpy()
            
        num_samples = list(embeddings_map.values())[0].shape[0]
        ordered_keys = [k for k in ['mri', 'ecg', 'tabular', 'cine'] if k in keys]
        combined_emb = np.vstack([embeddings_map[k] for k in ordered_keys])
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(combined_emb)

        embedding_2d = embedding_2d - np.mean(embedding_2d, axis=0)

        coords_map = {}
        start = 0
        for key in ordered_keys:
            end = start + num_samples
            coords_map[key] = embedding_2d[start:end]
            start = end
            
        distances = {}
        perimeter = np.zeros(num_samples)
        
        # Calculate Pairwise Distances
        if 'mri' in ordered_keys and 'ecg' in ordered_keys:
            # Vectorized Euclidean distance between MRI and ECG points for all patients
            d = np.linalg.norm(coords_map['mri'] - coords_map['ecg'], axis=1)
            distances['mri_ecg'] = d
            perimeter += d
            
        if 'ecg' in ordered_keys and 'tabular' in ordered_keys:
            d = np.linalg.norm(coords_map['ecg'] - coords_map['tabular'], axis=1)
            distances['ecg_tab'] = d
            perimeter += d
            
        if 'tabular' in ordered_keys and 'mri' in ordered_keys:
            d = np.linalg.norm(coords_map['tabular'] - coords_map['mri'], axis=1)
            distances['tab_mri'] = d
            perimeter += d           
            
        if 'cine' in ordered_keys:
            if 'mri' in ordered_keys:
                d = np.linalg.norm(coords_map['cine'] - coords_map['mri'], axis=1)
                distances['cine_mri'] = d
                perimeter += d
            if 'ecg' in ordered_keys:
                d = np.linalg.norm(coords_map['cine'] - coords_map['ecg'], axis=1)
                distances['cine_ecg'] = d
                perimeter += d
            if 'tabular' in ordered_keys:
                d = np.linalg.norm(coords_map['cine'] - coords_map['tabular'], axis=1)
                distances['cine_tab'] = d
                perimeter += d
            
        # calculate averages
        avg_perimeter = np.mean(perimeter)
        
        # Log to WandB
        wandb_logger = trainer.logger.experiment
        log_data = {
            "val/umap_avg_perimeter": avg_perimeter,
            "epoch": self.current_epoch
        }
        
        # Log individual edge averages
        for pair_name, dist_array in distances.items():
            log_data[f"val/umap_dist_{pair_name}"] = np.mean(dist_array)
            
        wandb_logger.log(log_data)
        
        print(f"\n[Epoch {self.current_epoch}] UMAP Alignment Stats:")
        print(f"  Avg subject circumference: {avg_perimeter:.4f}")
        for k, v in distances.items():
            print(f"  Avg {k}: {np.mean(v):.4f}")
     
       
        fig, ax = plt.subplots(figsize=(12, 10))
        
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal', 'box')


        styles = {
            'mri': {'color': 'blue', 'marker': 'o', 'label': 'MRI'},
            'ecg': {'color': 'red', 'marker': 'x', 'label': 'ECG'},
            'tabular': {'color': 'green', 'marker': '^', 'label': 'Tabular'},
            'cine': {'color': 'purple', 'marker': 's', 'label': 'Cine'},
        }
        
        
        for key in ordered_keys:
            s = styles[key]
            ax.scatter(coords_map[key][:, 0], coords_map[key][:, 1], 
                       c=s['color'], alpha=0.1, s=2, marker=s['marker'], label=f"{s['label']} (All)")
            
    
        num_viz = min(20, num_samples)
        indices = np.random.choice(num_samples, num_viz, replace=False)
        
        for idx in indices:
            points = {k: coords_map[k][idx] for k in ordered_keys}
            
            # Plot Dots
            for key, pt in points.items():
                s = styles[key]                
                if s['marker'] == 'x':
                    ax.scatter(pt[0], pt[1], c=s['color'], s=40, marker=s['marker'], linewidths=2)
                else:
                    # 'o' and '^' support edgecolors
                    ax.scatter(pt[0], pt[1], c=s['color'], s=30, edgecolors='black', marker=s['marker'])
            
            # Draw Lines
            # If 3 modalities: Triangle
            if 'mri' in points and 'ecg' in points:
                ax.plot([points['mri'][0], points['ecg'][0]], [points['mri'][1], points['ecg'][1]], 'k--', alpha=0.3)
            if 'ecg' in points and 'tabular' in points:
                ax.plot([points['ecg'][0], points['tabular'][0]], [points['ecg'][1], points['tabular'][1]], 'k--', alpha=0.3)
            if 'tabular' in points and 'mri' in points:
                ax.plot([points['tabular'][0], points['mri'][0]], [points['tabular'][1], points['mri'][1]], 'k--', alpha=0.3)
            if 'cine' in points:
                if 'mri' in points:
                    ax.plot([points['cine'][0], points['mri'][0]], [points['cine'][1], points['mri'][1]], 'k--', alpha=0.3)
                if 'ecg' in points:
                    ax.plot([points['cine'][0], points['ecg'][0]], [points['cine'][1], points['ecg'][1]], 'k--', alpha=0.3)
                if 'tabular' in points:
                    ax.plot([points['cine'][0], points['tabular'][0]], [points['cine'][1], points['tabular'][1]], 'k--', alpha=0.3)

        # Legend
        from matplotlib.lines import Line2D
        custom_lines = []
        labels = []
        for key in ordered_keys:
            s = styles[key]
            custom_lines.append(Line2D([0], [0], color=s['color'], lw=4, marker=s['marker'], linestyle='None'))
            labels.append(s['label'])
            
            
            
        custom_lines.append(Line2D([0], [0], color='black', linestyle='--', lw=1))
        labels.append('Same Patient Link')
        
        ax.legend(custom_lines, labels, loc='upper right')
        
        ax.set_title(f"Alignment: {' + '.join([styles[k]['label'] for k in ordered_keys])} (Epoch {self.current_epoch})")
        
        stats_text = (
            f"Avg Perimeter: {avg_perimeter:.2f}\n"
            f"-----------------\n"
        )
        for k, v in distances.items():
            stats_text += f"{k}: {np.mean(v):.2f}\n"

        # Place text box in top-left corner
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, fontfamily='monospace')
        wandb_logger = trainer.logger.experiment
        wandb_logger.log({"UMAP_Subject_Alignment": wandb.Image(fig)})
        plt.close(fig)

        self.val_preds_storage.clear()

    def log_umap_plots_to_wandb_coloured(self, trainer, is_best_epoch=False):
        """Uses pure UMAP to visualize alignment and biomarker overlays."""
        if getattr(self, 'UMAP_AVAILABLE', True) is False:
            self.val_preds_storage.clear()
            return
            
        if not self.val_preds_storage:
            return
            
        # Determine if we should plot this epoch
        should_plot = (self.current_epoch == 0) or (self.current_epoch % 10 == 0) or is_best_epoch
        if not should_plot:
            self.val_preds_storage.clear()
            return
        
        keys = list(self.val_preds_storage[0].keys())
        ordered_keys = [k for k in ['mri', 'cine', 'ecg', 'tabular'] if k in keys]
        
        embeddings_map = {}
        for key in ordered_keys:
            embeddings_map[key] = torch.cat([x[key] for x in self.val_preds_storage], dim=0).cpu().numpy()
            
        labels_concat = None
        if 'labels' in keys:
            labels_concat = torch.cat([x['labels'] for x in self.val_preds_storage], dim=0).cpu().numpy()
            
        num_samples = list(embeddings_map.values())[0].shape[0]
        
        # denormalize labels
        lvef_vals, lvm_vals = None, None
        if labels_concat is not None and hasattr(self, 'target_mean') and hasattr(self, 'target_std'):
            mean_np = self.target_mean.cpu().numpy()
            std_np = self.target_std.cpu().numpy()
            # LVEF is 11 and LVM is 13
            try:
                lvef_vals = labels_concat[:, 11] * std_np[11] + mean_np[11]
                lvm_vals = labels_concat[:, 13] * std_np[13] + mean_np[13]
            except IndexError:
                pass
        
        #
        combined_emb = np.vstack([embeddings_map[k] for k in ordered_keys])
        print(f"Running UMAP for alignment visualization ({combined_emb.shape[0]} points)...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
        embedding_2d = reducer.fit_transform(combined_emb)
        
    
        embedding_2d = embedding_2d - np.mean(embedding_2d, axis=0)

        coords_map = {}
        start = 0
        for key in ordered_keys:
            end = start + num_samples
            coords_map[key] = embedding_2d[start:end]
            start = end
            
        # calculate perimeter 
        distances = {}
        perimeter = np.zeros(num_samples)
        def calc_dist(k1, k2):
            if k1 in coords_map and k2 in coords_map:
                d = np.linalg.norm(coords_map[k1] - coords_map[k2], axis=1)
                distances[f"{k1}_{k2}"] = d
                return d
            return 0.0

        p_mri_ecg = calc_dist('mri', 'ecg')
        p_ecg_tab = calc_dist('ecg', 'tabular')
        p_tab_mri = calc_dist('tabular', 'mri')
        perimeter += p_mri_ecg + p_ecg_tab + p_tab_mri
        avg_perimeter = np.mean(perimeter)

        # Log Stats to WandB
        if hasattr(trainer.logger, 'experiment'):
            wandb_logger = trainer.logger.experiment
            log_data = {"val/umap_avg_perimeter": avg_perimeter, "epoch": self.current_epoch}
            for k, v in distances.items():
                log_data[f"val/umap_dist_{k}"] = np.mean(v)
            wandb_logger.log(log_data)
            
        
        def generate_plot(plot_type, biomarker_vals=None, title_suffix="", vmin=None, vmax=None):
            styles = {
                'mri':     {'color': '#1f77b4', 'marker': 'o', 'label': 'MRI (Localizer)'},
                'cine':    {'color': '#9467bd', 'marker': 's', 'label': 'Cine MRI'},
                'ecg':     {'color': '#d62728', 'marker': '^', 'label': 'ECG'},
                'tabular': {'color': '#2ca02c', 'marker': 'D', 'label': 'Tabular'},
            }

            fig, ax = plt.subplots(figsize=(14, 10))
            x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
            y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
            pad = (x_max - x_min) * 0.05
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_aspect('equal', 'box')
            ax.axis('off')

            scatters = []
            for key in ordered_keys:
                s = styles[key]
                if plot_type == 'standard':
                    c = s['color']
                    cmap = None
                else:
                    c = biomarker_vals
                    cmap = 'plasma'

                sc = ax.scatter(coords_map[key][:, 0], coords_map[key][:, 1], 
                           c=c, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, s=30, marker=s['marker'], 
                           edgecolors='white', linewidths=0.5, zorder=5)
                if plot_type != 'standard': scatters.append(sc)

            # Highlighted Connections (100 links)
            indices = np.random.choice(num_samples, min(100, num_samples), replace=False)
            for idx in indices:
                def draw_line(k1, k2):
                    if k1 in coords_map and k2 in coords_map:
                        ax.plot([coords_map[k1][idx, 0], coords_map[k2][idx, 0]], 
                                [coords_map[k1][idx, 1], coords_map[k2][idx, 1]], 
                                color='gray', linestyle='-', linewidth=0.5, alpha=0.3, zorder=1)
                draw_line('mri', 'ecg')
                draw_line('ecg', 'tabular')
                draw_line('tabular', 'mri')

           
            from matplotlib.lines import Line2D
            custom_lines = []
            labels = []
            for key in ordered_keys:
                s = styles[key]
                line = Line2D([0], [0], color='white', markerfacecolor='gray' if plot_type != 'standard' else s['color'], 
                              marker=s['marker'], markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                custom_lines.append(line)
                labels.append(s['label'])
                
            ax.legend(custom_lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, edgecolor='lightgray', fontsize=10)
            
            # Colorbar
            if plot_type != 'standard' and scatters:
                cbar = plt.colorbar(scatters[0], ax=ax, fraction=0.03, pad=0.04)
                unit = "(g)" if "LVM" in plot_type else "(%)"
                cbar.set_label(f'{plot_type} {unit}', rotation=270, labelpad=20, fontweight='bold', fontsize=12)

            # Title & Stats Box
            epoch_str = "Best Checkpoint" if is_best_epoch else f"Epoch {self.current_epoch}"
            ax.set_title(f"Multi-modal UMAP - {epoch_str} {title_suffix}", fontsize=16, fontweight='bold', pad=20)
            
            stats_text = f"Avg Perimeter: {avg_perimeter:.2f}\n" + "-" * 20 + "\n"
            for k, v in distances.items():
                stats_text += f"{k}: {np.mean(v):.2f}\n"
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
            fig.text(0.82, 0.6, stats_text, fontsize=9, verticalalignment='top', bbox=props, fontfamily='monospace')
            plt.subplots_adjust(right=0.8)
            
            return fig

        # Create all three plots
        figs = {}
        figs['standard'] = generate_plot('standard')
        if lvef_vals is not None:
             figs['LVEF'] = generate_plot('LVEF', lvef_vals, title_suffix="(Colored by LVEF)")
        if lvm_vals is not None:
           
             clamped_lvm = np.clip(lvm_vals, 50, 200)
             figs['LVM'] = generate_plot('LVM', clamped_lvm, title_suffix="(Colored by LVM)", vmin=50, vmax=200)

       
        save_dir = os.path.join(trainer.default_root_dir, "umap_plots")
        os.makedirs(save_dir, exist_ok=True)
        epoch_str = "best" if is_best_epoch else f"epoch_{self.current_epoch:03d}"
        
        if hasattr(trainer.logger, 'experiment'):
             log_dict = {f"UMAP_Alignment_{name}": wandb.Image(fig) for name, fig in figs.items()}
             trainer.logger.experiment.log(log_dict)

        for name, fig in figs.items():
            filename = f"umap_{name}_{epoch_str}"
            fig.savefig(os.path.join(save_dir, f"{filename}.svg"), format='svg', bbox_inches='tight')
            fig.savefig(os.path.join(save_dir, f"{filename}.png"), format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        print(f"Saved {len(figs)} UMAP plots to {save_dir}")
        self.val_preds_storage.clear()
    
    def log_tsne_plots_to_wandb_(self, trainer):
        if not self.val_preds_storage:
            return
        

        keys = list(self.val_preds_storage[0].keys())
        embeddings_map = {}
        for key in keys:
            embeddings_map[key] = torch.cat([x[key] for x in self.val_preds_storage], dim=0).cpu().numpy()
            
        num_samples = list(embeddings_map.values())[0].shape[0]

        ordered_keys = [k for k in ['mri', 'cine', 'ecg', 'tabular'] if k in keys]
        

        combined_emb = np.vstack([embeddings_map[k] for k in ordered_keys])
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(combined_emb)


        embedding_2d = embedding_2d - np.mean(embedding_2d, axis=0)

        coords_map = {}
        start = 0
        for key in ordered_keys:
            end = start + num_samples
            coords_map[key] = embedding_2d[start:end]
            start = end

        distances = {}
        perimeter = np.zeros(num_samples)
        
        def calc_dist(k1, k2):
            if k1 in coords_map and k2 in coords_map:
                d = np.linalg.norm(coords_map[k1] - coords_map[k2], axis=1)
                distances[f"{k1}_{k2}"] = d
                return d
            return 0.0


        p_mri_ecg = calc_dist('mri', 'ecg')
        p_ecg_tab = calc_dist('ecg', 'tabular')
        p_tab_mri = calc_dist('tabular', 'mri')
        perimeter += p_mri_ecg + p_ecg_tab + p_tab_mri
        
        if 'cine' in coords_map:
            perimeter += calc_dist('cine', 'mri')
            perimeter += calc_dist('cine', 'ecg')
            perimeter += calc_dist('cine', 'tabular')
            
        avg_perimeter = np.mean(perimeter)
        

        wandb_logger = trainer.logger.experiment
        log_data = {"val/umap_avg_perimeter": avg_perimeter, "epoch": self.current_epoch}
        for k, v in distances.items():
            log_data[f"val/umap_dist_{k}"] = np.mean(v)
        wandb_logger.log(log_data)
        
        print(f"\n[Epoch {self.current_epoch}] UMAP Stats | Avg Perimeter: {avg_perimeter:.4f}")

        styles = {
            'mri':     {'color': '#1f77b4', 'marker': 'o', 'label': 'MRI (Localizer)'},  # Blue
            'cine':    {'color': '#9467bd', 'marker': 's', 'label': 'Cine MRI'},         # Purple
            'ecg':     {'color': '#d62728', 'marker': '^', 'label': 'ECG'},              # Red
            'tabular': {'color': '#2ca02c', 'marker': 'D', 'label': 'Tabular'},          # Green
        }

        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set limits with padding
        x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
        y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
        pad = 2.0
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect('equal', 'box')
        ax.axis('off')  # Hide axis 

        for key in ordered_keys:
            s = styles[key]
            ax.scatter(coords_map[key][:, 0], coords_map[key][:, 1], 
                       c=s['color'], alpha=0.08, s=10, marker=s['marker'], 
                       edgecolors='none', label=None) 

        num_viz = min(20, num_samples)
        indices = np.random.choice(num_samples, num_viz, replace=False)
        
        for idx in indices:
            points = {k: coords_map[k][idx] for k in ordered_keys}
            
            def draw_line(k1, k2):
                if k1 in points and k2 in points:
                    ax.plot([points[k1][0], points[k2][0]], 
                            [points[k1][1], points[k2][1]], 
                            color='black', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)

           
            draw_line('mri', 'ecg')
            draw_line('ecg', 'tabular')
            draw_line('tabular', 'mri')
            
            if 'cine' in points:
                draw_line('cine', 'mri')
                draw_line('cine', 'ecg')
                draw_line('cine', 'tabular')

            for key, pt in points.items():
                s = styles[key]
                ax.scatter(pt[0], pt[1], c=s['color'], s=80, 
                           marker=s['marker'], edgecolors='white', linewidth=1.5, zorder=10)

        from matplotlib.lines import Line2D
        custom_lines = []
        labels = []
        
        for key in ordered_keys:
            s = styles[key]
            # Create proxy artist for legend
            line = Line2D([0], [0], color='white', markerfacecolor=s['color'], 
                          marker=s['marker'], markersize=10, markeredgecolor='white', markeredgewidth=1.5)
            custom_lines.append(line)
            labels.append(s['label'])
            
        custom_lines.append(Line2D([0], [0], color='black', linestyle='-', linewidth=1, alpha=0.6))
        labels.append('Patient Link')
        
        ax.legend(custom_lines, labels, loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=10)
        
        ax.set_title(f"Multi-modal Alignment Manifold (Epoch {self.current_epoch})", fontsize=14, fontweight='bold', pad=10)
        
        # Stats Box
        stats_text = f"Avg Perimeter: {avg_perimeter:.2f}\n" + "-" * 20 + "\n"
        for k, v in distances.items():
            stats_text += f"{k}: {np.mean(v):.2f}\n"

        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='lightgray')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, fontfamily='monospace')


        wandb_logger.log({"UMAP_Subject_Alignment": wandb.Image(fig)})
        
        # 
        # save svg on first epoch, and every 10 epochs
        should_save_vector = (self.current_epoch == 0) or (self.current_epoch % 10 == 0)
        
        if should_save_vector:
            save_dir = os.path.join(trainer.default_root_dir, "umap_plots")
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"umap_alignment_epoch_{self.current_epoch:03d}"
            # Save SVG (Vector - Best for Papers)
            plt.savefig(os.path.join(save_dir, f"{filename}.svg"), format='svg', bbox_inches='tight')
            # Save PNG (High Res - Best for Slides)
            plt.savefig(os.path.join(save_dir, f"{filename}.png"), format='png', dpi=300, bbox_inches='tight')
            print(f"Saved publication-ready UMAP plots to {save_dir}")

        plt.close(fig)
        self.val_preds_storage.clear()
        
    #TSNE version to bypass UMAP Numba errors in some environments
    def log_umap_plots_to_wandb_tsne(self, trainer):
        """Uses t-SNE (sklearn) to visualize alignment without Numba dependencies."""
        if not self.val_preds_storage:
            return
        
        # aggregate Data
        keys = list(self.val_preds_storage[0].keys())
        embeddings_map = {}
        for key in keys:
            embeddings_map[key] = torch.cat([x[key] for x in self.val_preds_storage], dim=0).cpu().numpy()
            
        num_samples = list(embeddings_map.values())[0].shape[0]
        ordered_keys = [k for k in ['mri', 'cine', 'ecg', 'tabular'] if k in keys]
        
        # dimensionality reduction (t-SNE)
        combined_emb = np.vstack([embeddings_map[k] for k in ordered_keys])
        
        print(f"Running t-SNE for alignment visualization ({combined_emb.shape[0]} points)...")
        reducer = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
        embedding_2d = reducer.fit_transform(combined_emb)

       
        embedding_2d = embedding_2d - np.mean(embedding_2d, axis=0)

        coords_map = {}
        start = 0
        for key in ordered_keys:
            end = start + num_samples
            coords_map[key] = embedding_2d[start:end]
            start = end
            
        #
        distances = {}
        perimeter = np.zeros(num_samples)
        
        def calc_dist(k1, k2):
            if k1 in coords_map and k2 in coords_map:
                d = np.linalg.norm(coords_map[k1] - coords_map[k2], axis=1)
                distances[f"{k1}_{k2}"] = d
                return d
            return 0.0

        p_mri_ecg = calc_dist('mri', 'ecg')
        p_ecg_tab = calc_dist('ecg', 'tabular')
        p_tab_mri = calc_dist('tabular', 'mri')
        perimeter += p_mri_ecg + p_ecg_tab + p_tab_mri
        
        if 'cine' in coords_map:
            perimeter += calc_dist('cine', 'mri')
            perimeter += calc_dist('cine', 'ecg')
            perimeter += calc_dist('cine', 'tabular')
            
        avg_perimeter = np.mean(perimeter)
        
        # Log Stats
        if hasattr(trainer.logger, 'experiment'):
            wandb_logger = trainer.logger.experiment
            log_data = {"val/umap_avg_perimeter": avg_perimeter, "epoch": self.current_epoch}
            for k, v in distances.items():
                log_data[f"val/umap_dist_{k}"] = np.mean(v)
            wandb_logger.log(log_data)
        
        print(f"\n[Epoch {self.current_epoch}] Alignment Stats | Avg Perimeter: {avg_perimeter:.4f}")

        
        styles = {
            'mri':     {'color': '#1f77b4', 'marker': 'o', 'label': 'MRI (Localizer)'},
            'cine':    {'color': '#9467bd', 'marker': 's', 'label': 'Cine MRI'},
            'ecg':     {'color': '#d62728', 'marker': '^', 'label': 'ECG'},
            'tabular': {'color': '#2ca02c', 'marker': 'D', 'label': 'Tabular'},
        }

        fig, ax = plt.subplots(figsize=(14, 10))
        
        x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
        y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
        pad = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect('equal', 'box')
        ax.axis('off')

  
        for key in ordered_keys:
            s = styles[key]
            ax.scatter(coords_map[key][:, 0], coords_map[key][:, 1], 
                       c=s['color'], alpha=0.08, s=10, marker=s['marker'], 
                       edgecolors='none') 

      
        num_viz = min(20, num_samples)
        indices = np.random.choice(num_samples, num_viz, replace=False)
        
        for idx in indices:
            points = {k: coords_map[k][idx] for k in ordered_keys}
            
            def draw_line(k1, k2):
                if k1 in points and k2 in points:
                    ax.plot([points[k1][0], points[k2][0]], 
                            [points[k1][1], points[k2][1]], 
                            color='black', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)

            draw_line('mri', 'ecg')
            draw_line('ecg', 'tabular')
            draw_line('tabular', 'mri')
            if 'cine' in points:
                draw_line('cine', 'mri')
                draw_line('cine', 'ecg')
                draw_line('cine', 'tabular')

            for key, pt in points.items():
                s = styles[key]
                ax.scatter(pt[0], pt[1], c=s['color'], s=80, 
                           marker=s['marker'], edgecolors='white', linewidth=1.5, zorder=10)

       
        from matplotlib.lines import Line2D
        custom_lines = []
        labels = []
        for key in ordered_keys:
            s = styles[key]
            line = Line2D([0], [0], color='white', markerfacecolor=s['color'], 
                          marker=s['marker'], markersize=10, markeredgecolor='white', markeredgewidth=1.5)
            custom_lines.append(line)
            labels.append(s['label'])
            
        custom_lines.append(Line2D([0], [0], color='black', linestyle='-', linewidth=1, alpha=0.6))
        labels.append('Patient Link')
        
        ax.legend(custom_lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1), 
                  frameon=True, edgecolor='lightgray', title="Modalities", fontsize=10)
        
        ax.set_title(f"Multi-modal Alignment (t-SNE) - Epoch {self.current_epoch}", 
                     fontsize=16, fontweight='bold', pad=20)
        
       
        stats_text = f"Avg Perimeter: {avg_perimeter:.2f}\n" + "-" * 20 + "\n"
        for k, v in distances.items():
            stats_text += f"{k}: {np.mean(v):.2f}\n"

    
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
        fig.text(0.82, 0.6, stats_text, fontsize=9, verticalalignment='top', bbox=props, fontfamily='monospace')

        # Adjust layout to make room for the external legend/text
        plt.subplots_adjust(right=0.8)

        if hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"UMAP_Subject_Alignment": wandb.Image(fig)})
        
        should_save = (self.current_epoch == 0) or (self.current_epoch % 10 == 0)
        if should_save:
            save_dir = os.path.join(trainer.default_root_dir, "alignment_plots")
            os.makedirs(save_dir, exist_ok=True)
            filename = f"alignment_epoch_{self.current_epoch:03d}"
            plt.savefig(os.path.join(save_dir, f"{filename}.svg"), format='svg', bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f"{filename}.png"), format='png', dpi=300, bbox_inches='tight')
            print(f"Saved plots to {save_dir}")

        plt.close(fig)
        self.val_preds_storage.clear()