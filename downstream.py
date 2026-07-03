import argparse
from calendar import c
from re import U
import os
for key in list(os.environ.keys()):
    if key.startswith("SLURM_"):
        del os.environ[key]
from sympy import im
import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.plugins.environments import LightningEnvironment
import os
from dataclasses import dataclass, field
from typing import List
# Import your project modules
from src.data.trimodal_config import TriModalConfig as Config
from src.data.TrimodalDatamodule import MultimodalDataModule
from src.models.Downstream  import TriModalDownstreamRegressor
torch.set_float32_matmul_precision('high')

BIOMARKER_COLUMNS = [
    'LAV max (mL)', 'LAV min (mL)', 'LASV (mL)', 'LAEF (%)',
    'RAV max (mL)', 'RAV min (mL)', 'RASV (mL)', 'RAEF (%)',
    'LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)', 'LVEF (%)',
    'LVCO (L/min)', 'LVM (g)',
    'RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)'
]
@dataclass
class ModelConfig:
    embedding_dim: int = 384
    encoder_num_heads: int = 6
    encoder_num_layers: int = 12
    encoder_mlp_ratio: int = 4
    decoder_depth: int = 4
    decoder_embed_dim: int = 192
    mask_ratio: float = 0.75
    grad_checkpointing: bool = False

@dataclass
class DataConfig:
    data_path: str = ""
    train_ids_path: str = ""
    val_ids_path: str = ""
    test_ids_path: str = ""
    batch_size: int = 64
    num_workers: int = 4
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)

@dataclass
class TrainConfig:
    experiment_name: str = ""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 120
    warmup_epochs: int = 5
    log_interval: int = 10
    wandb_project: str = ""
    seed: int = 42

@dataclass
class GlobalConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
def main():
    # 1. Argument Parsing for Flexible Experiments
    parser = argparse.ArgumentParser(description="Downstream Regression Task")
    
    # Critical Paths
    parser.add_argument("--upstream_ckpt", type=str, default=None, help="Path to the pre-trained contrastive checkpoint (.ckpt)")
    
    
    parser.add_argument("--only_test", action="store_true", help="Skip training and run only the test phase")
    parser.add_argument("--test_ckpt", type=str, default=None, help="Path to a trained downstream checkpoint (.ckpt) for testing. Required if --only_test is set.")
    # Ablation Controls
    parser.add_argument("--modalities", nargs='+', default=None, #['mri', 'ecg', 'tabular'], 
                        choices=['mri', 'localizer_12s', 'ecg', 'tabular', 'cine', 'cine_12f','cine_12f_9s', 'cine_24f'], 
                        help="Which modalities to use for regression? (e.g. --modalities mri ecg)")
    
    #low-data regime flags (optional)
    parser.add_argument("--use_10_percent", action="store_true", help="Use only 10% of the training data (stratified sampling)")
    parser.add_argument("--use_1_percent", action="store_true", help="Use only 1% of the training data (stratified sampling)")
    
    parser.add_argument("--freeze", action="store_true", help="If set, freezes the upstream encoder (Linear Probing). If not, fine-tunes.")
    
    # Training Overrides
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for downstream task")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for downstream task")
    parser.add_argument("--epochs", type=int, default=120, help="Max epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_log", action="store_true", help="Disable WandB logging")
    parser.add_argument("--run_name", type=str, default=None, help="Optional custom run name for logging and checkpoints")

    args = parser.parse_args()
    
    # setup
    pl.seed_everything(args.seed)
    
    # initialize Config
    config = Config()
    config.batch_size = args.batch_size # Override batch size
    
    if args.use_10_percent:
        print("Using 10% of training data (Stratified Sample)")
        config.train_ids_path = config.train_ids_path_10_percent
    elif args.use_1_percent:
        print("Using 1% of training data (Stratified Sample)")
        config.train_ids_path = config.train_ids_path_1_percent
    else:
        print("Using full training data")
        
        
    config.use_localizer_12s = 'localizer_12s' in args.modalities
    config.use_mri = 'mri' in args.modalities or config.use_localizer_12s
    config.use_ecg = 'ecg' in args.modalities
    config.use_tabular = 'tabular' in args.modalities
       
    config.use_cine_12_frames = 'cine_12f' in args.modalities
    config.use_cine_12f_9s = 'cine_12f_9s' in args.modalities
    config.use_cine_24_frames = 'cine_24f' in args.modalities
    # Force use_cine to True so the Dataset appends it to the batch
    config.use_cine = 'cine' in args.modalities or config.use_cine_12_frames or config.use_cine_24_frames or config.use_cine_12f_9s
    print(f"\Experiment Config:")
    print(f"Modalities: {args.modalities}")
    print(f"Upstream Ckpt: {args.upstream_ckpt if args.upstream_ckpt else 'Scratch (Random Init)'}")
    print(f"Strategy: {'Linear Probe (Frozen)' if args.freeze else 'Fine-Tuning'}")
    
    
    # data Module
    print("Initializing DataModule...")
    dm = MultimodalDataModule(config=config) # 
    dm.setup(stage="fit") # run setup manually to calculate stats before model init
    vocab_sizes = {}
    target_mean = None
    target_std = None
    if (config.use_tabular):
        vocab_sizes = dm.vocab_sizes
        print(f"Vocab Sizes Detected: {vocab_sizes}")
        config.numerical_features = config.numerical_features_basic

    target_mean = dm.target_mean
    target_std = dm.target_std
    # initialize Downstream Model
    
    
    dm.setup(stage="test")
    if dm.test_ds is not None:
        test_ids = dm.test_ds.patient_ids
        print(f"\n[INFO] Final Test Set Size: {len(test_ids)}")
             
        # Save to .pt file
        torch.save(test_ids, f"final_test_ids_{len(test_ids)}.pt")
        print(f"[INFO] Saved test IDs to 'final_test_ids_{len(test_ids)}.pt'\n")
    else:
        print("\nWarning: Test dataset is empty or None!\n")
        
        

    model = TriModalDownstreamRegressor(
        upstream_config=config,
        vocab_sizes=vocab_sizes,
        target_mean=target_mean,
        target_std=target_std,
        upstream_checkpoint_path=args.upstream_ckpt if not args.only_test else None,
        active_modalities=args.modalities,
        freeze_encoder=args.freeze,
        lr=args.lr,
        # Pass biomarker names for logging
        biomarker_names=BIOMARKER_COLUMNS
    )
    model = torch.compile(model)

    # loggers & Callbacks
    args.upstream_ckpt = args.test_ckpt if args.only_test else args.upstream_ckpt
    upstream_ckpt_name = args.upstream_ckpt.split('/')[-2].replace('.ckpt', '') if args.upstream_ckpt else 'scratch'
    run_name = args.run_name if args.run_name else f"Downstream_{'_'.join(args.modalities)}_{upstream_ckpt_name}_{'frozen' if args.freeze else 'finetune'}_data_{'10percent' if args.use_10_percent else '1percent' if args.use_1_percent else 'full'}"
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/downstream",
            filename=f"{run_name}_{{epoch:02d}}_{{val/loss:.4f}}",
            monitor="val/loss",
            mode="min",
            save_top_k=1
        ),
        EarlyStopping(monitor="val/loss", patience=20, mode="min", verbose=True),
        LearningRateMonitor(logging_interval='epoch')
    ]

    if not args.no_log:
        wandb_logger = WandbLogger(
            project="Cardiac-Synergy-Downstream",
            name=f"{run_name}",
            log_model=False
        )
    else:
        wandb_logger = None
            # trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        accumulate_grad_batches=4,
        precision="16-mixed",  
        plugins=[LightningEnvironment()],
    )



    if args.only_test:
        print("\n" + "="*30)
        print(" TEST MODE ONLY ")
        print("="*30)
        
        if not args.test_ckpt:
            raise ValueError("Error: --test_ckpt must be provided when using --only_test")
            
        print(f"Loading trained weights from: {args.test_ckpt}")
        checkpoint = torch.load(args.test_ckpt, map_location=model.device, weights_only=False)
        # load state dict
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        trainer.test(model, dm)
    else:
        
        

    # train & Test
        trainer.fit(model, dm)
    
        print("\nStarting Testing on Best Model...")
        
        # manually load the best checkpoint with safety override
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            print(f"Loading best checkpoint manually from: {best_path}")
            
            # load checkpoint dictionary safely
            checkpoint = torch.load(best_path, map_location=model.device, weights_only=False)
            
            # load weights into the existing model
            # use strict=False because Lightning checkpoints sometimes have extra keys
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            
            # run test (without passing ckpt_path, so it uses the currently loaded model)
            trainer.test(model, dm)
        else:
            print("Warning: No best checkpoint found. Testing with current weights.")
            trainer.test(model, dm)
        

if __name__ == "__main__":
    main()