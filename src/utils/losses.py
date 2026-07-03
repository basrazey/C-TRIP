"""
Loss Functions for Tri-Modal Contrastive Learning and Biomarker Prediction

Includes:
1. TriModalContrastiveLoss - Contrastive loss for 3 modalities
2. Weighted regression losses for biomarker prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TriModalContrastiveLoss(nn.Module):
    """
    Tri-modal contrastive loss inspired by CLIP and Synergy-CLIP.
    
    Computes contrastive loss for 3 pairwise comparisons:
    1. Localizer ↔ ECG
    2. ECG ↔ Tabular
    3. Tabular ↔ Localizer
    
    Each comparison is bidirectional (modality A → modality B and B → A).
    """
    
    def __init__(self,
                 lambda_localizer_ecg: float = 1.0,
                 lambda_ecg_tabular: float = 1.0,
                 lambda_tabular_localizer: float = 1.0):
        """
        Args:
            lambda_localizer_ecg: Weight for localizer-ECG loss
            lambda_ecg_tabular: Weight for ECG-tabular loss
            lambda_tabular_localizer: Weight for tabular-localizer loss
        """
        super().__init__()
        self.lambda_localizer_ecg = lambda_localizer_ecg
        self.lambda_ecg_tabular = lambda_ecg_tabular
        self.lambda_tabular_localizer = lambda_tabular_localizer
        self.ce_loss = nn.CrossEntropyLoss()
    
    def compute_pairwise_loss(self, 
                             similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute bidirectional contrastive loss for a similarity matrix.
        
        Args:
            similarity_matrix: [batch, batch] similarity matrix (already temperature-scaled)
        
        Returns:
            Scalar loss
        """
        batch_size = similarity_matrix.shape[0]
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Loss in both directions
        loss_forward = self.ce_loss(similarity_matrix, labels)
        loss_backward = self.ce_loss(similarity_matrix.T, labels)
        
        # Average
        loss = (loss_forward + loss_backward) / 2.0
        return loss
    
    def forward(self, similarities: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute tri-modal contrastive loss.
        
        Args:
            similarities: Dict with keys 'localizer_ecg', 'ecg_tabular', 'tabular_localizer'
                         Each value is a [batch, batch] similarity matrix
        
        Returns:
            Dict with 'total_loss' and individual losses
        """
        # Compute loss for each pairwise comparison
        loss_localizer_ecg = self.compute_pairwise_loss(similarities['localizer_ecg'])
        loss_ecg_tabular = self.compute_pairwise_loss(similarities['ecg_tabular'])
        loss_tabular_localizer = self.compute_pairwise_loss(similarities['tabular_localizer'])
        
        # Weighted sum
        total_loss = (
            self.lambda_localizer_ecg * loss_localizer_ecg +
            self.lambda_ecg_tabular * loss_ecg_tabular +
            self.lambda_tabular_localizer * loss_tabular_localizer
        )
        
        return {
            'total_loss': total_loss,
            'loss_localizer_ecg': loss_localizer_ecg,
            'loss_ecg_tabular': loss_ecg_tabular,
            'loss_tabular_localizer': loss_tabular_localizer
        }


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss for multi-task biomarker prediction.
    Allows prioritizing important biomarkers (e.g., LVEF, RVEF, LVM).
    """
    
    def __init__(self, 
                 weights: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        Args:
            weights: [num_biomarkers] tensor of loss weights
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.weights = weights
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch, num_biomarkers]
            targets: [batch, num_biomarkers]
        
        Returns:
            Scalar loss
        """
        # Compute MSE per biomarker
        mse = (predictions - targets) ** 2  # [batch, num_biomarkers]
        
        # Apply weights if provided
        if self.weights is not None:
            weights = self.weights.to(mse.device)
            mse = mse * weights.unsqueeze(0)  # Broadcast weights
        
        # Reduction
        if self.reduction == 'mean':
            return mse.mean()
        elif self.reduction == 'sum':
            return mse.sum()
        else:
            return mse


class WeightedHuberLoss(nn.Module):
    """
    Weighted Huber loss (smooth L1 loss) for robust biomarker prediction.
    Less sensitive to outliers than MSE.
    """
    
    def __init__(self,
                 weights: Optional[torch.Tensor] = None,
                 delta: float = 1.0,
                 reduction: str = 'mean'):
        """
        Args:
            weights: [num_biomarkers] tensor of loss weights
            delta: Threshold for switching between L1 and L2
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.weights = weights
        self.delta = delta
        self.reduction = reduction
        self.huber = nn.SmoothL1Loss(reduction='none', beta=delta)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch, num_biomarkers]
            targets: [batch, num_biomarkers]
        
        Returns:
            Scalar loss
        """
        # Compute Huber loss per biomarker
        huber_loss = self.huber(predictions, targets)  # [batch, num_biomarkers]
        
        # Apply weights if provided
        if self.weights is not None:
            weights = self.weights.to(huber_loss.device)
            huber_loss = huber_loss * weights.unsqueeze(0)
        
        # Reduction
        if self.reduction == 'mean':
            return huber_loss.mean()
        elif self.reduction == 'sum':
            return huber_loss.sum()
        else:
            return huber_loss


def get_biomarker_weights(important_biomarkers: list = None,
                          num_biomarkers: int = 18,
                          important_weight: float = 2.0,
                          default_weight: float = 1.0) -> torch.Tensor:
    """
    Create weight tensor for biomarker prediction.
    Give higher weights to clinically important biomarkers.
    
    Args:
        important_biomarkers: List of indices for important biomarkers
                             Default: [3, 9, 5] for LVEF (%), RVEF (%), LVM (g)
        num_biomarkers: Total number of biomarkers
        important_weight: Weight for important biomarkers
        default_weight: Weight for other biomarkers
    
    Returns:
        [num_biomarkers] weight tensor
    """
    if important_biomarkers is None:
        # Default: LVEF (idx 3), LVM (idx 5), RVEF (idx 9)
        # Based on biomarker list:
        # 0:LVEDV, 1:LVESV, 2:LVSV, 3:LVEF, 4:LVCO, 5:LVM,
        # 6:RVEDV, 7:RVESV, 8:RVSV, 9:RVEF,
        # 10:LAV max, 11:LAV min, 12:LASV, 13:LAEF,
        # 14:RAV max, 15:RAV min, 16:RASV, 17:RAEF
        important_biomarkers = [3, 5, 9]  # LVEF, LVM, RVEF
    
    weights = torch.ones(num_biomarkers) * default_weight
    for idx in important_biomarkers:
        if idx < num_biomarkers:
            weights[idx] = important_weight
    
    return weights


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")
    print("=" * 80)
    
    # Test TriModalContrastiveLoss
    print("\n1. Testing TriModalContrastiveLoss")
    print("-" * 80)
    
    batch_size = 8
    projection_dim = 768
    
    # Create dummy similarity matrices
    localizer_emb = F.normalize(torch.randn(batch_size, projection_dim), dim=-1)
    ecg_emb = F.normalize(torch.randn(batch_size, projection_dim), dim=-1)
    tabular_emb = F.normalize(torch.randn(batch_size, projection_dim), dim=-1)
    
    temperature = 0.07
    similarities = {
        'localizer_ecg': torch.matmul(localizer_emb, ecg_emb.T) / temperature,
        'ecg_tabular': torch.matmul(ecg_emb, tabular_emb.T) / temperature,
        'tabular_localizer': torch.matmul(tabular_emb, localizer_emb.T) / temperature
    }
    
    contrastive_loss = TriModalContrastiveLoss(
        lambda_localizer_ecg=1.0,
        lambda_ecg_tabular=1.0,
        lambda_tabular_localizer=1.0
    )
    
    loss_dict = contrastive_loss(similarities)
    
    print(f"✓ Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"✓ Localizer-ECG loss: {loss_dict['loss_localizer_ecg'].item():.4f}")
    print(f"✓ ECG-Tabular loss: {loss_dict['loss_ecg_tabular'].item():.4f}")
    print(f"✓ Tabular-Localizer loss: {loss_dict['loss_tabular_localizer'].item():.4f}")
    
    # Test WeightedMSELoss
    print("\n2. Testing WeightedMSELoss")
    print("-" * 80)
    
    num_biomarkers = 18
    predictions = torch.randn(batch_size, num_biomarkers)
    targets = torch.randn(batch_size, num_biomarkers)
    
    # Test without weights
    mse_loss = WeightedMSELoss()
    loss = mse_loss(predictions, targets)
    print(f"✓ MSE loss (unweighted): {loss.item():.4f}")
    
    # Test with weights
    weights = get_biomarker_weights(important_biomarkers=[3, 5, 9])
    mse_loss_weighted = WeightedMSELoss(weights=weights)
    loss_weighted = mse_loss_weighted(predictions, targets)
    print(f"✓ MSE loss (weighted): {loss_weighted.item():.4f}")
    print(f"✓ Weights: {weights[:6].tolist()} ... (showing first 6)")
    
    # Test WeightedHuberLoss
    print("\n3. Testing WeightedHuberLoss")
    print("-" * 80)
    
    huber_loss = WeightedHuberLoss(weights=weights, delta=1.0)
    loss_huber = huber_loss(predictions, targets)
    print(f"✓ Huber loss (weighted): {loss_huber.item():.4f}")
    
    # Test with outliers
    predictions_outlier = predictions.clone()
    predictions_outlier[0, 0] = 100.0  # Add outlier
    
    loss_mse_outlier = mse_loss(predictions_outlier, targets)
    loss_huber_outlier = huber_loss(predictions_outlier, targets)
    
    print(f"\n✓ With outlier:")
    print(f"  - MSE loss: {loss_mse_outlier.item():.4f} (sensitive to outliers)")
    print(f"  - Huber loss: {loss_huber_outlier.item():.4f} (robust to outliers)")
    
    print(f"\n{'=' * 80}")
    print("✅ All loss function tests passed!")
    
    # Test biomarker weights
    print("\n4. Biomarker Weight Configuration")
    print("-" * 80)
    
    biomarker_names = [
        'LVEDV', 'LVESV', 'LVSV', 'LVEF*', 'LVCO', 'LVM*',
        'RVEDV', 'RVESV', 'RVSV', 'RVEF*',
        'LAV max', 'LAV min', 'LASV', 'LAEF',
        'RAV max', 'RAV min', 'RASV', 'RAEF'
    ]
    
    weights = get_biomarker_weights()
    print("Biomarker weights (* = important):")
    for i, (name, weight) in enumerate(zip(biomarker_names, weights)):
        print(f"  {i:2d}. {name:12s}: {weight:.1f}")
