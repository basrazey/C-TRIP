"""
Metrics Tracker for biomarker prediction evaluation
"""

import torch
import numpy as np
from typing import Dict, Optional
from scipy.stats import pearsonr


BIOMARKER_NAMES = [
    'LAV max (mL)', 'LAV min (mL)', 'LASV (mL)', 'LAEF (%)',
    'RAV max (mL)', 'RAV min (mL)', 'RASV (mL)', 'RAEF (%)',
    'LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)', 'LVEF (%)',
    'LVCO (L/min)', 'LVM (g)',
    'RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)'
]


class MetricsTracker:
    """
    Track and compute evaluation metrics for biomarker prediction.
    
    Metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R²: Coefficient of determination
    - Pearson: Pearson correlation coefficient
    """
    
    def __init__(self, biomarker_names: Optional[list] = None):
        """
        Args:
            biomarker_names: List of biomarker names (default: standard 18)
        """
        self.biomarker_names = biomarker_names or BIOMARKER_NAMES
        self.reset()
    
    def reset(self):
        """Reset all tracked values."""
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update with new batch of predictions.
        
        Args:
            predictions: [batch, num_biomarkers]
            targets: [batch, num_biomarkers]
        """
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
    
    def compute(self) -> Dict:
        """
        Compute all metrics.
        
        Returns:
            Dict with 'average' and 'per_biomarker' metrics
        """
        if not self.predictions:
            return {}
        
        # Concatenate all batches
        pred = torch.cat(self.predictions, dim=0).numpy()  # [N, num_biomarkers]
        target = torch.cat(self.targets, dim=0).numpy()  # [N, num_biomarkers]
        
        # Per-biomarker metrics
        per_biomarker = {}
        mae_list = []
        rmse_list = []
        r2_list = []
        pearson_list = []
        
        for i, name in enumerate(self.biomarker_names):
            pred_i = pred[:, i]
            target_i = target[:, i]
            
            # MAE
            mae = np.mean(np.abs(pred_i - target_i))
            
            # RMSE
            rmse = np.sqrt(np.mean((pred_i - target_i) ** 2))
            
            # R²
            ss_res = np.sum((target_i - pred_i) ** 2)
            ss_tot = np.sum((target_i - np.mean(target_i)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Pearson correlation
            try:
                pearson_corr, _ = pearsonr(pred_i, target_i)
            except:
                pearson_corr = 0.0
            
            per_biomarker[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Pearson': pearson_corr
            }
            
            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)
            pearson_list.append(pearson_corr)
        
        # Average metrics
        average = {
            'MAE': np.mean(mae_list),
            'RMSE': np.mean(rmse_list),
            'R2': np.mean(r2_list),
            'Pearson': np.mean(pearson_list)
        }
        
        return {
            'average': average,
            'per_biomarker': per_biomarker
        }


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """
    Pretty print metrics.
    
    Args:
        metrics: Output from MetricsTracker.compute()
        title: Title for the output
    """
    if not metrics:
        print("No metrics to display")
        return
    
    print("\n" + "=" * 100)
    print(f"{title:^100}")
    print("=" * 100)
    
    # Average metrics
    if 'average' in metrics:
        avg = metrics['average']
        print(f"\n{'AVERAGE METRICS':^100}")
        print("-" * 100)
        print(f"{'Metric':<20} {'Value':>15}")
        print("-" * 100)
        print(f"{'MAE':<20} {avg['MAE']:>15.4f}")
        print(f"{'RMSE':<20} {avg['RMSE']:>15.4f}")
        print(f"{'R²':<20} {avg['R2']:>15.4f}")
        print(f"{'Pearson':<20} {avg['Pearson']:>15.4f}")
    
    # Per-biomarker metrics
    if 'per_biomarker' in metrics:
        print(f"\n{'PER-BIOMARKER METRICS':^100}")
        print("-" * 100)
        print(f"{'Biomarker':<30} {'MAE':>12} {'RMSE':>12} {'R²':>12} {'Pearson':>12}")
        print("-" * 100)
        
        for name, m in metrics['per_biomarker'].items():
            print(f"{name:<30} {m['MAE']:>12.4f} {m['RMSE']:>12.4f} {m['R2']:>12.4f} {m['Pearson']:>12.4f}")
    
    print("=" * 100 + "\n")


if __name__ == "__main__":
    print("Testing MetricsTracker...")
    print("=" * 80)
    
    # Create tracker
    tracker = MetricsTracker()
    
    # Simulate some predictions
    for _ in range(5):
        pred = torch.randn(8, 18)
        target = pred + torch.randn(8, 18) * 0.1  # Close to predictions
        tracker.update(pred, target)
    
    # Compute metrics
    metrics = tracker.compute()
    
    # Print
    print_metrics(metrics, title="Test Metrics")
    
    print("✅ MetricsTracker test passed!")