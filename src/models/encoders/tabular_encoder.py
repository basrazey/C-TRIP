"""
Tabular Encoder - Transformer-based encoder for patient tabular data
Adapted from ViTa's tabular encoder 

Handles three types of features:
1. Numerical features (e.g., age, BMI, blood pressure)
2. Binary categorical features (e.g., sex, diabetes diagnosis)
3. Multi-categorical features (e.g., smoking status, alcohol intake)

Input: Dict with 'numerical', 'binary_categorical', 'multi_categorical' tensors
Output: Embedding tensor [batch, embed_dim]
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List


class TabularEncoder(nn.Module):
    """
    Transformer-based encoder for tabular data.
    Accepts raw tensor [B, num_features] and processes internally.
    """
    
    def __init__(self,
                 input_size: int,
                 embed_dim: int = 768,
                 depth: int = 4,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 output_type: str = 'cls'):
        """
        Args:
            input_size: Number of input features
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            output_type: 'cls' or 'mean'
        """
        super().__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.output_type = output_type
        

        self.feature_projection = nn.Linear(input_size, embed_dim)
        
        # CLS token
        if output_type == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings
        num_tokens = 1 + (1 if output_type == 'cls' else 0)  # 1 feature token + optional CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(depth)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.output_type == 'cls':
            nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, num_features] tensor (preprocessed, NaNs handled)
        
        Returns:
            [batch, embed_dim] embedding
        """
       
        features = torch.nan_to_num(features, nan=0.0)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        batch_size = features.shape[0]
        
        # Project features: [B, num_features] -> [B, 1, embed_dim]
        x = self.feature_projection(features).unsqueeze(1)
        
        # Add CLS token if needed
        if self.output_type == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Normalize
        x = self.norm(x)
        
        # Extract output
        if self.output_type == 'cls':
            output = x[:, 0]  # CLS token
        else:
            output = x.mean(dim=1)  # Mean pooling
        
        return output

# UK Biobank Feature Configuration (from ViTa Appendix D)


# Numerical features (67 features)
UKBB_NUMERICAL_FEATURES = [
    'Pulse wave Arterial Stiffness index-2.0',
    'Systolic blood pressure-2.mean',
    'Diastolic blood pressure-2.mean',
    'Pulse rate-2.mean',
    'Body fat percentage-2.0',
    'Whole body fat mass-2.0',
    'Whole body fat-free mass-2.0',
    'Whole body water mass-2.0',
    'Body mass index (BMI)-2.0',
    'Cooked vegetable intake-2.0',
    'Salad / raw vegetable intake-2.0',
    'Cardiac operations performed',
    'Total mass-2.0',
    'Basal metabolic rate-2.0',
    'Impedance of whole body-2.0',
    'Waist circumference-2.0',
    'Hip circumference-2.0',
    'Standing height-2.0',
    'Height-2.0',
    'Sitting height-2.0',
    'Weight-2.0',
    'Ventricular rate-2.0',
    'P duration-2.0',
    'QRS duration-2.0',
    'PQ interval-2.0',
    'RR interval-2.0',
    'PP interval-2.0',
    'Cardiac output-2.0',
    'Cardiac index-2.0',
    'Average heart rate-2.0',
    'Body surface area-2.0',
    'Duration of walks-2.0',
    'Duration of moderate activity-2.0',
    'Duration of vigorous activity-2.0',
    'Time spent watching television (TV)-2.0',
    'Time spent using computer-2.0',
    'Time spent driving-2.0',
    'Heart rate during PWA-2.0',
    'Systolic brachial blood pressure during PWA-2.0',
    'Diastolic brachial blood pressure during PWA-2.0',
    'Peripheral pulse pressure during PWA-2.0',
    'Central systolic blood pressure during PWA-2.0',
    'Central pulse pressure during PWA-2.0',
    'Number of beats in waveform average for PWA-2.0',
    'Central augmentation pressure during PWA-2.0',
    'Augmentation index for PWA-2.0',
    'Cardiac output during PWA-2.0',
    'End systolic pressure during PWA-2.0',
    'End systolic pressure index during PWA-2.0',
    'Stroke volume during PWA-2.0',
    'Mean arterial pressure during PWA-2.0',
    'Cardiac index during PWA-2.0',
    'Sleep duration-2.0',
    'Exposure to tobacco smoke at home-2.0',
    'Exposure to tobacco smoke outside home-2.0',
    'Pack years of smoking-2.0',
    'Pack years adult smoking as proportion of life span exposed to smoking-2.0',
    # Biomarkers (18 cardiac phenotypes) - these are targets, not inputs for pretraining
    # But can be used as inputs in some scenarios where CMR is used as aux. modality
    'LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)', 'LVEF (%)', 'LVCO (L/min)', 'LVM (g)',
    'RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)',
    'LAV max (mL)', 'LAV min (mL)', 'LASV (mL)', 'LAEF (%)',
    'RAV max (mL)', 'RAV min (mL)', 'RASV (mL)', 'RAEF (%)'
]

# Binary categorical features (18 features)
UKBB_BINARY_FEATURES = [
    'Worrier / anxious feelings-2.0',
    'Shortness of breath walking on level ground-2.0',
    'Sex-0.0',
    'Diabetes diagnosis',
    'Heart attack diagnosed by doctor',
    'Angina diagnosed by doctor',
    'Stroke diagnosed by doctor',
    'High blood pressure diagnosed by doctor',
    'Cholesterol lowering medication regularly taken',
    'Blood pressure medication regularly taken',
    'Insulin medication regularly taken',
    'Hormone replacement therapy medication regularly taken',
    'Oral contraceptive pill or minipill medication regularly taken',
    'Pace-maker-2.0',
    'Ever had diabetes (Type I or Type II)-0.0',
    'Long-standing illness, disability or infirmity-2.0',
    "Tense / 'highly strung'-2.0",
    'Ever smoked-2.0'
]

# Multi-categorical features (name: num_classes)
UKBB_MULTI_CATEGORICAL_FEATURES = {
    'Sleeplessness / insomnia-2.0': 3,
    'Frequency of heavy DIY in last 4 weeks-2.0': 7,
    'Alcohol intake frequency.-2.0': 6,
    'Processed meat intake-2.0': 6,
    'Beef intake-2.0': 6,
    'Pork intake-2.0': 6,
    'Lamb/mutton intake-2.0': 6,
    'Overall health rating-2.0': 4,
    'Alcohol usually taken with meals-2.0': 3,
    'Alcohol drinker status-2.0': 3,
    'Frequency of drinking alcohol-0.0': 5,
    'Frequency of consuming six or more units of alcohol-0.0': 5,
    'Amount of alcohol drunk on a typical drinking day-0.0': 6,
    'Falls in the last year-2.0': 3,
    'Weight change compared with 1 year ago-2.0': 3,
    'Number of days/week walked 10+ minutes-2.0': 8,
    'Number of days/week of moderate physical activity 10+ minutes-2.0': 8,
    'Number of days/week of vigorous physical activity 10+ minutes-2.0': 8,
    'Usual walking pace-2.0': 3,
    'Frequency of stair climbing in last 4 weeks-2.0': 6,
    'Frequency of walking for pleasure in last 4 weeks-2.0': 7,
    'Duration walking for pleasure-2.0': 8,
    'Frequency of strenuous sports in last 4 weeks-2.0': 7,
    'Duration of strenuous sports-2.0': 8,
    'Duration of light DIY-2.0': 8,
    'Duration of heavy DIY-2.0': 8,
    'Frequency of other exercises in last 4 weeks-2.0': 7,
    'Duration of other exercises-2.0': 8,
    'Current tobacco smoking-2.0': 3,
    'Past tobacco smoking-2.0': 4,
    'Smoking/smokers in household-2.0': 3,
    'Smoking status-2.0': 3
}


def build_tabular_encoder(embed_dim=768,
                          depth=4,
                          num_heads=8,
                          input_size=233,
                          include_biomarkers=False,
                          pretrained_path=None,
                          **kwargs):
    """
    Build tabular encoder.
    
    Args:
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        input_size: Number of input features (auto-detected from data)
        include_biomarkers: Not used (biomarkers are separate targets)
        pretrained_path: Path to pretrained weights
    
    Returns:
        TabularEncoder model
    """
    model = TabularEncoder(
        input_size=input_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
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
        
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded tabular pretrained weights from {pretrained_path}")
        if msg.missing_keys:
            print(f"Missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"Unexpected keys: {msg.unexpected_keys}")
    
    return model

