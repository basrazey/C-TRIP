import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger
from sympy import im
from wandb import Config # Optional
import torch
from src.data.trimodal_config import TriModalConfig as Config
torch.serialization.add_safe_globals([Config])
from src.data.TrimodalDatamodule import MultimodalDataModule
from src.models.TriModalSynergyCLIP import TriModalSynergyCLIP
import os
import pandas as pd
import numpy as np
import argparse
import src
import wandb
import datetime
torch.set_float32_matmul_precision('high')
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="TriModal Training")
    
    # Modality Flags (Default is False to force explicit selection)
    parser.add_argument("--use_localizer_12s", action="store_true", help="Use new 12-slice localizer")
    parser.add_argument("--mri_ckpt", type=str, default=None, help="Path to MAE encoder_only.pt")
    parser.add_argument("--cine_ckpt", type=str, default=None, help="Path to Cine MAE encoder_only.pt")
    parser.add_argument("--use_ecg", action="store_true")
    parser.add_argument("--use_tabular", action="store_true")
    parser.add_argument("--use_pheno", action="store_true")
    parser.add_argument("--use_cine", action="store_true", help="Use standard 3-slice cine")
    parser.add_argument("--use_cine_12_frames", action="store_true", help="Use 12-frame, 3-slice (36 channel) cine")
    
    # Overrides
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no_ckpt", action="store_true", help="Start from scratch (no pre-trained weights)")
    parser.add_argument("--aug_mode", type=str, default='soft', choices=['soft', 'hard'])
    
    #temp override arguments for loss ablation
    parser.add_argument("--temp_mri_ecg", type=float, default=None, help="Override temperature for MRI-ECG loss")
    parser.add_argument("--temp_mri_tab", type=float, default=None, help="Override temperature for MRI-Tabular loss")
    parser.add_argument("--temp_ecg_tab", type=float, default=None, help="Override temperature for ECG-Tabular loss")   
    parser.add_argument("--temp_cine_mri", type=float, default=None, help="Override temperature for Cine-MRI loss")
    parser.add_argument("--temp_cine_tab", type=float, default=None, help="Override temperature for Cine-Tabular loss")
    parser.add_argument("--temp_cine_ecg", type=float, default=None, help="Override temperature for Cine-ECG loss")
    
    #loss wweight overrides
    parser.add_argument("--w_cine_mri", type=float, default=1.0)
    parser.add_argument("--w_cine_tab", type=float, default=1.0)
    parser.add_argument("--w_cine_ecg", type=float, default=1.0)
    
    parser.add_argument("--w_mri_ecg", type=float, default=1.0)
    parser.add_argument("--w_mri_tab", type=float, default=1.0)
    parser.add_argument("--w_ecg_tab", type=float, default=1.0)
    

    #continue from checkpoint
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to continue training from")
    
    args = parser.parse_args()
    
    # 1. Init Config
    config = Config()
    pl.seed_everything(42)
    config.use_mri = True # Localizer is ALWAYS active
    config.use_localizer_12s = args.use_localizer_12s
    config.use_ecg = args.use_ecg
    config.use_tabular = args.use_tabular
    config.use_cine_12_frames = args.use_cine_12_frames
    config.use_cine = args.use_cine or args.use_cine_12_frames

    if args.mri_ckpt:
        config.mri_pretrained_ckpt_path = args.mri_ckpt
    
    if args.cine_ckpt:
        config.cine_pretrained_ckpt_path = args.cine_ckpt

    if args.batch_size: config.batch_size = args.batch_size
    if args.lr: config.learning_rate = args.lr
    config.aug_mode = args.aug_mode
    
    if args.no_ckpt:
        print("Training from scratch (Pre-trained weights disabled)")
        config.mri_pretrained_ckpt_path = None
        config.ecg_pretrained_ckpt_path = None
        config.tabular_pretrained_ckpt_path = None
        config.cine_pretrained_ckpt_path = None
        
    if args.temp_mri_ecg: config.temp_mri_ecg = args.temp_mri_ecg
    if args.temp_mri_tab: config.temp_mri_tab = args.temp_mri_tab
    if args.temp_ecg_tab: config.temp_ecg_tab = args.temp_ecg_tab
    if args.temp_cine_mri: config.temp_cine_mri = args.temp_cine_mri
    if args.temp_cine_tab: config.temp_cine_tab = args.temp_cine_tab
    if args.temp_cine_ecg: config.temp_cine_ecg = args.temp_cine_ecg
    
    if args.w_cine_mri: config.w_cine_mri = args.w_cine_mri
    if args.w_cine_tab: config.w_cine_tab = args.w_cine_tab
    if args.w_cine_ecg: config.w_cine_ecg = args.w_cine_ecg
    
    if args.w_mri_ecg: config.w_mri_ecg = args.w_mri_ecg
    if args.w_mri_tab: config.w_mri_tab = args.w_mri_tab
    if args.w_ecg_tab: config.w_ecg_tab = args.w_ecg_tab
    
    run_tags = ["Localizer"]
    losses_active = []

    # Case: Localizer + Cine + Tabular
    if config.use_cine and config.use_tabular and not config.use_ecg:
        print("\nLocalizer + Cine + Tabular")
        print("tabular: Using phenotypes")
        config.numerical_features = config.numerical_features_phenotype
        config.tabular_pretrained_ckpt_path = config.tabular_pretrained_ckpt_path_with_pheno
        
        # Loss Config
        config.use_ecg_tab_loss = False 
        losses_active = ["Loc-Cine", "Loc-Tab", "Cine-Tab"]
        run_name = "Loc_Cine_Tab(Pheno)"

    # Case: Localizer + ECG + Tabular
    elif config.use_ecg and config.use_tabular and not config.use_cine:
        if args.use_pheno:
            print("\nLocalizer + ECG + Tabular")
            print("Tabular: WITH Phenotypes")
            config.numerical_features = config.numerical_features_phenotype
            config.tabular_pretrained_ckpt_path = config.tabular_pretrained_ckpt_path_with_pheno
            
            # Loss Config bimodal only: Focus on Localizer-ECG and Localizer-Tabular alignment. ECG-Tabular loss is noisy and less critical for synergy.
            config.use_ecg_tab_loss = False
            losses_active = ["Loc-ECG", "Loc-Tab"]
            run_name = "Loc_ECG_Tab(Pheno)"
            
        else:
            print("\nLocalizer + ECG + Tabular")
            print("Tabular: No Phenotypes")
            config.numerical_features = config.numerical_features_basic
            config.tabular_pretrained_ckpt_path = config.tabular_pretrained_ckpt_path_basic
            
            # Loss Config bimodal only: Focus on Localizer-ECG and Localizer-Tabular alignment. ECG-Tabular loss is noisy and less critical for synergy.
            config.use_ecg_tab_loss = False
            losses_active = ["Loc-ECG", "Loc-Tab"]
            run_name = "Loc_ECG_Tab(Basic)"

    # Case: Localizer + Cine + ECG
    elif config.use_cine and config.use_ecg and not config.use_tabular:
        print("\nLocalizer + Cine + ECG")
        # Loss Config: Trimodal (Triangle Closed)
        losses_active = ["Loc-Cine", "Loc-ECG", "Cine-ECG"]
        run_name = "Loc_Cine_ECG"

    # Case: Simple Pairs
    elif config.use_cine:
        run_name = "Loc_Cine_Pair"
        losses_active = ["Loc-Cine"]
    elif config.use_ecg:
        run_name = "Loc_ECG_Pair"
        losses_active = ["Loc-ECG"]
    elif config.use_tabular:
        if args.use_pheno:
            print("\nLocalizer + ECG + Tabular")
            print("Tabular: WITH Phenotypes")
            config.numerical_features = config.numerical_features_phenotype
            config.tabular_pretrained_ckpt_path = config.tabular_pretrained_ckpt_path_with_pheno
            
            losses_active = ["Loc-Tab"]
            run_name = "Loc_Tab(Pheno)"
            
        else:
            print("\nLocalizer + Tabular")
            print("Tabular: Using BASIC features (Default for pair)")
            config.numerical_features = config.numerical_features_basic
            config.tabular_pretrained_ckpt_path = config.tabular_pretrained_ckpt_path_basic
            run_name = "Loc_Tab_Pair"
            losses_active = ["Loc-Tab"]
    else:
        raise ValueError("Error: Localizer needs at least one other modality (Cine, ECG, or Tabular).")

    run_name = run_name +"_tempMRITAB"+str(config.temp_mri_tab)
    print(f"Active Losses: {', '.join(losses_active)}")
    
    
    
    

    # 2. Initialize DataModule
    print("Initializing DataModule...")
    dm = MultimodalDataModule(config)
    
    #run setup() manually first to calculate vocab sizes
    dm.setup(stage='fit') 
    vocab_sizes = dm.vocab_sizes if config.use_tabular else {}
    if (config.use_tabular):
        vocab_sizes = dm.vocab_sizes
        print(f"Vocab Sizes Detected: {vocab_sizes}")
        print(f"Input Feature Count: {len(config.numerical_features)} numerical")
        


    # 4. Initialize Model
    print("Initializing Model...")
    model = TriModalSynergyCLIP(
        config=config, 
        vocab_sizes=vocab_sizes,
        target_mean=dm.target_mean,
        target_std=dm.target_std,
        mri_chkpt_path=config.mri_pretrained_ckpt_path,
        ecg_chkpt_path=config.ecg_pretrained_ckpt_path,
        tabular_chkpt_path=config.tabular_pretrained_ckpt_path,
        cine_chkpt_path=config.cine_pretrained_ckpt_path,
    )
    #model = torch.compile(model)
    
    
    callbacks = [LearningRateMonitor(logging_interval='step')]
    
    loss_mapping = {
        "Loc-Cine": "val/loss_cine_mri",
        "Loc-Tab":  "val/loss_mri_tab",
        "Loc-ECG":  "val/loss_mri_ecg",
        "Cine-Tab": "val/loss_cine_tab",
        "Cine-ECG": "val/loss_cine_ecg"
    }
    for loss_name in losses_active:
        metric = loss_mapping.get(loss_name)
        if metric:
            callbacks.append(EarlyStopping(
                monitor=metric, patience=20, mode="min", verbose=True, 
                check_on_train_epoch_end=False
            ))
            print(f"Added EarlyStopping for {loss_name}")
            
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    run_id = f"{run_name}_{timestamp}"
            
    checkpoint_cb = ModelCheckpoint(
        dirpath='checkpoints/',
        monitor='val/loss_total',
        mode='min',
        filename=f'{run_id}-{{epoch:02d}}-{{val/loss_total:.3f}}',
        save_top_k=1
    )
    callbacks.append(checkpoint_cb)



    wandb_logger = WandbLogger(project="Cardiac-Synergy-CLIP", name=run_id, log_model=False)
        
        
    trainer = pl.Trainer(
        max_epochs=150,
        accelerator='gpu',
        devices=1, 
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        #accumulate_grad_batches=config.accumulate_grad_batches,
        precision=config.precision,
    )
    
    print("Starting Training...")
    
    if args.ckpt:
        print(f"Resuming training from checkpoint: {args.ckpt}")
        trainer.fit(model, dm, ckpt_path=args.ckpt)
    else:
        trainer.fit(model, dm)  
    #log where ckpt is saved    
    print(f"Best Checkpoint saved at: {checkpoint_cb.best_model_path}")
    
    if checkpoint_cb.best_model_path:
        print("Generating final t-SNE plots for the best checkpoint...")
        # Load the best weights
        checkpoint = torch.load(checkpoint_cb.best_model_path, map_location=model.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Run a validation epoch manually to collect embeddings
        trainer.validate(model, datamodule=dm)
        

        model.log_umap_plots_to_wandb_coloured(trainer, is_best_epoch=True)
        
    wandb.finish()
    
    