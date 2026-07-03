from calendar import c
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import wandb
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Dict, Optional
import datetime
from pathlib import Path

from src.utils.bland_altman_multiple import main_single_run

# import upstream model
from src.models.TriModalSynergyCLIP import TriModalSynergyCLIP

class MultimodalTransformerFusion(nn.Module):
    def __init__(self, mri_dim=768, ecg_dim=384, hidden_dim=512, num_heads=8, num_layers=2):
        super().__init__()
        # Project both modalities to the same dimension
        self.mri_proj = nn.Linear(mri_dim, hidden_dim)
        self.ecg_proj = nn.Linear(ecg_dim, hidden_dim)
        
        # Learned modality embeddings 
        self.modality_embeds = nn.Parameter(torch.randn(1, 2, hidden_dim))
        
        #  Transformer block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            batch_first=True, 
            activation="gelu",
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, mri, ecg):
        # reshape from (Batch, Dim) -> (Batch, 1, Hidden_Dim)
        mri_emb = self.mri_proj(mri).unsqueeze(1) 
        ecg_emb = self.ecg_proj(ecg).unsqueeze(1) 
        
        # cmbine into a sequence of length 2: (Batch, 2, Hidden_Dim)
        seq = torch.cat([mri_emb, ecg_emb], dim=1)
        
        # add the modality positional embeddings
        seq = seq + self.modality_embeds
        
        attended_seq = self.transformer(seq)
        
        # flatten back to 1D vector to pass into regression MLP
        # Output size: Batch x (hidden_dim * 2)
        return attended_seq.flatten(start_dim=1)
    
    

class TriModalDownstreamRegressor(pl.LightningModule):
    """
    Downstream Regression Module for Cardiac Biomarkers.
    
    Features:
    - Supports Ablation: Select specific modalities via `active_modalities`.
    - Supports Modes: Linear Probing (frozen encoder) or Fine-Tuning.
    """
    def __init__(
        self,
        upstream_config,
        vocab_sizes: Dict,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        upstream_checkpoint_path: Optional[str] = None,
        active_modalities: List[str] = ['mri', 'localizer_12s', 'ecg', 'tabular', 'cine'], 
        freeze_encoder: bool = True,
        num_biomarkers: int = 18,
        biomarker_names: List[str] = None,
        biomarker_stats: Dict = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['target_mean', 'target_std'])
        
        self.active_modalities = active_modalities
        self.biomarker_names = biomarker_names or [f'bio_{i}' for i in range(num_biomarkers)]
        self.biomarker_stats =  {'mean': target_mean.clone(), 'std': target_std.clone()} #dict from target_mean and target_std
        
        self.has_cine = any(c in active_modalities for c in ['cine', 'cine_12f', 'cine_12f_9s', 'cine_24f'])
        self.has_mri = any(m in active_modalities for m in ['mri', 'localizer_12s'])
        print(f" in trimodal regressor: Active Modalities: {active_modalities}")
        
        upstream_config.use_localizer_12s = 'localizer_12s' in active_modalities
        upstream_config.use_mri = self.has_mri
        print(f"Upstream Config - use_localizer_12s: {upstream_config.use_localizer_12s}, use_mri: {upstream_config.use_mri}")
        
        upstream_config.use_cine_12_frames = 'cine_12f' in active_modalities
        upstream_config.use_cine_12f_9s = 'cine_12f_9s' in active_modalities
        upstream_config.use_cine_24_frames = 'cine_24f' in active_modalities
        # Force it to True so the Upstream Encoder processes it!
        upstream_config.use_cine = self.has_cine
        
        upstream_config.use_ecg = 'ecg' in active_modalities
        upstream_config.use_tabular = 'tabular' in active_modalities

        # initialize upstream model 
        # initialize the full model structure
        self.upstream_model = TriModalSynergyCLIP(
            config=upstream_config,
            vocab_sizes=vocab_sizes,
            target_mean=target_mean,
            target_std=target_std,
            mri_chkpt_path=None # load the full weights below
        )
        
        # load weights (if pretraining) or leave random (if scratch)
        if upstream_checkpoint_path:
            print(f"Loading Upstream Weights from: {upstream_checkpoint_path}")

            checkpoint = torch.load(upstream_checkpoint_path, map_location='cpu', weights_only=False)
            
        
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            # Load weights (strict=False to ignore the old projection heads )
            missing, unexpected = self.upstream_model.load_state_dict(state_dict, strict=False)
            print(f"Weights Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            print("Initializing Encoders from SCRATCH (Random Init)")

        # Freeze or unfreeze
        if freeze_encoder:
            print("Freezing Upstream Encoders (Linear Probing)")
            self.upstream_model.eval()
            for param in self.upstream_model.parameters():
                param.requires_grad = False
        else:
            print("Fine-Tuning Encoders Enabled")

        # Construct regression head dynamically
        # calculate concatenation size based on active modalities
        
        
        # initialize fusion block if both MRI and ECG are active
        self.use_deep_fusion = self.has_mri and 'ecg' in active_modalities
        if self.use_deep_fusion:
            print("Using Transformer Cross-Attention Fusion for MRI + ECG")
            self.fusion_block = MultimodalTransformerFusion(
                mri_dim=768, 
                ecg_dim=getattr(self.upstream_model.ecg_encoder, 'embed_dim', 384),
                hidden_dim=512, 
                num_layers=2
            )
            # fusion block outputs 2 tokens of size 512, flattened to 1024
            self.input_dim = 1024 
        else:
            # (keep  old input_dim calculation for other baselines)
            self.input_dim = 0
            if self.has_mri: self.input_dim += 768
            if 'ecg' in active_modalities: self.input_dim += getattr(self.upstream_model.ecg_encoder, 'embed_dim', 384)
            if 'tabular' in active_modalities: self.input_dim += getattr(upstream_config, 'tab_embed_dim', 384)
            if self.has_cine: self.input_dim += 768
        
        print(f"constructing Regression Head for {active_modalities}. Input Dim: {self.input_dim}")

        self.head = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_biomarkers)
        )

        # loss Setup
        self.criterion = nn.MSELoss()
        
        # optional: weighted loss for difficult targets (LVEF/RVEF)
        self.register_buffer('target_weights', torch.ones(num_biomarkers))
        if biomarker_names:
            for i, name in enumerate(biomarker_names):
                if 'LVEF' in name or 'RVEF' in name:
                    self.target_weights[i] = 1.0  # change here for weighting

  
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, mri=None, ecg=None, tabular=None, cine=None):
        # features from upstream 
        # pass None for modalities to ignore in the ablation
        
        # control inputs based on active_modalities
        _mri = mri if self.has_mri else None
        _ecg = ecg if 'ecg' in self.active_modalities else None
        _tab = tabular if 'tabular' in self.active_modalities else None
        _cine = cine if self.has_cine else None
        
        # Upstream forward returns (mri_feat, ecg_feat, tab_feat)
        with torch.set_grad_enabled(not self.hparams.freeze_encoder):
            mri_feat, ecg_feat, tab_feat, cine_feat = self.upstream_model(_mri, _ecg, _tab, _cine)
        
        
        
        if self.use_deep_fusion:
            # use  Transformer to fuse them
            combined_features = self.fusion_block(mri_feat, ecg_feat)
        else:
            # concatenate active features
            features_list = []
            if self.has_mri: features_list.append(mri_feat)
            if 'ecg' in self.active_modalities: features_list.append(ecg_feat)
            if 'tabular' in self.active_modalities: features_list.append(tab_feat)
            if self.has_cine: features_list.append(cine_feat)
            
            if not features_list:
                raise ValueError("No active modalities provided to forward pass!")
            
            combined_features = torch.cat(features_list, dim=1)
        
    
        return self.head(combined_features)

    def training_step(self, batch, batch_idx):
        mri = batch.get('mri')
        ecg = batch.get('ecg')
        tabular = batch.get('tabular')
        cine = batch.get('cine')
        targets = batch.get('label')
        
        preds = self(mri, ecg, tabular, cine)
        
        # weighted MSE (currently 1 for all targets, but can be adjusted for difficult targets)
        loss = (self.criterion(preds, targets) * self.target_weights).mean()
        
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mri = batch.get('mri')
        ecg = batch.get('ecg')
        tabular = batch.get('tabular')
        cine = batch.get('cine')
        targets = batch.get('label')
        ids = batch.get('id')
        
        preds = self(mri, ecg, tabular, cine)
        
        loss = (self.criterion(preds, targets) * self.target_weights).mean()
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        
        self.validation_step_outputs.append({
            'preds': preds.detach().cpu(),
            'targets': targets.detach().cpu(),
            'ids': ids
        })
        
        
        return loss

    def test_step(self, batch, batch_idx):
        mri = batch.get('mri')
        ecg = batch.get('ecg')
        tabular = batch.get('tabular')
        cine = batch.get('cine')    
        targets = batch.get('label')
        
        ids = batch.get('id')

        preds = self(mri, ecg, tabular, cine)
        
        self.test_step_outputs.append({
            'preds': preds.detach().cpu(),
            'targets': targets.detach().cpu(),
            'ids': ids
        })

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        # concatenate all batches
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs], dim=0).detach().cpu()
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs], dim=0).detach().cpu()
        

        if self.biomarker_stats:
            mean = self.biomarker_stats['mean'].detach().cpu().numpy()
            std = self.biomarker_stats['std'].detach().cpu().numpy()
            all_preds_denorm = all_preds.numpy() * std + mean
            all_targets_denorm = all_targets.numpy() * std + mean
        else:
            print("!!!No biomarker stats found. Reporting Normalized metrics.")
            all_preds_denorm = all_preds.numpy()
            all_targets_denorm = all_targets.numpy()    
        
        # Compute Metrics
        metrics = self._compute_metrics(all_preds_denorm, all_targets_denorm, prefix='val')
        for k, v in metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        #save csv every 30 epochs
        if (self.current_epoch + 1) % 30 == 0:
            all_ids = []
            for x in self.validation_step_outputs:
                ids = x['ids']
                if isinstance(ids, torch.Tensor): ids = ids.cpu().tolist()
                all_ids.extend(ids)
            self._save_results(all_ids, all_preds_denorm, all_targets_denorm) 
        
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        if not self.test_step_outputs: return
        
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs], dim=0).detach().cpu()
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs], dim=0).detach().cpu()
        
        # handle IDs 
        all_ids = []
        for x in self.test_step_outputs:
            ids = x['ids']
            if isinstance(ids, torch.Tensor): ids = ids.cpu().tolist()
            all_ids.extend(ids)

        # Denormalize
        if self.biomarker_stats:
            mean = self.biomarker_stats['mean'].detach().cpu().numpy()
            std = self.biomarker_stats['std'].detach().cpu().numpy()
            all_preds = all_preds.numpy() * std + mean
            all_targets = all_targets.numpy() * std + mean
        else:
            all_preds = all_preds.numpy()
            all_targets = all_targets.numpy()
        
        # metrics
        metrics = self._compute_metrics(all_preds, all_targets, prefix='test')
        for k, v in metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        # save CSV
        self._save_results(all_ids, all_preds, all_targets)
        self.test_step_outputs.clear()

    def _denormalize(self, data):
        if self.biomarker_stats:
            mean = self.biomarker_stats['mean'].cpu().numpy()
            std = self.biomarker_stats['std'].cpu().numpy()
            return data * std + mean
        return data

    def _compute_metrics(self, preds, targets, prefix):
        metrics = {}
        maes, r2s = [], []
        
        for i, name in enumerate(self.biomarker_names):
            p = preds[:, i]
            t = targets[:, i]
            
            mae = mean_absolute_error(t, p)
            r2 = r2_score(t, p)
            
            clean_name = name.split('(')[0].strip().replace(" ", "_").replace("/", "_")
            metrics[f"{prefix}/{clean_name}_MAE"] = mae
            metrics[f"{prefix}/{clean_name}_R2"] = r2
            
            maes.append(mae)
            r2s.append(r2)
            
        metrics[f"{prefix}/avg_MAE"] = np.mean(maes)
        metrics[f"{prefix}/avg_R2"] = np.mean(r2s)
        return metrics

    def _save_results(self, ids, preds, targets):
        save_dir = Path("test_results")
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        data = {'PatientID': ids}
        for i, name in enumerate(self.biomarker_names):
            clean_name = name.replace(" ", "_")
            data[f"{clean_name}_Pred"] = preds[:, i]
            data[f"{clean_name}_GT"] = targets[:, i]
            
        df = pd.DataFrame(data)
        path = save_dir / f"predictions_{timestamp}.csv"
        df.to_csv(path, index=False)
        print(f"Saved test predictions to {path}")
        print("Running Bland-Altman Analysis on Test Predictions...")
        self._call_bland_altman(str(path))
        
    def _call_bland_altman(self, predictions_file):
        pred_timestamp = predictions_file.split('/')[-1].split('.')[0] #split pred files to take date "20260127_083502"predictions_file.split('_')[-1].split('.')[0]
        output_dir = f"./test_results/results_{pred_timestamp}/bland_altman_analysis"
        print(f"Running Bland-Altman Analysis for {predictions_file}. Output Dir: {output_dir}")
        biomarkers = None  # None = all biomarkers
        
        main_single_run(predictions_file, output_dir, biomarkers)
        
  

    def configure_optimizers(self):
        encoder_params = [p for p in self.upstream_model.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        """
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )"""
        
        param_groups = [{'params': head_params, 'lr': self.hparams.lr}]
        
        # only add the encoder group if fine-tuning
        if not self.hparams.freeze_encoder and len(encoder_params) > 0:
            param_groups.append({'params': encoder_params, 
                                 'lr': self.hparams.lr * 0.1})
            
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        
        
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.hparams.lr * 0.1, self.hparams.lr],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1, # 10% warmup
            cycle_momentum=False
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
                "frequency": 1
            }
        }