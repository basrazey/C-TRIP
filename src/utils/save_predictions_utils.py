"""
Utility functions for saving biomarker predictions and ground truth.

Saves results in the format:
Patient_ID, [Biomarker]_Predicted, [Biomarker]_GroundTruth, [Biomarker]_AbsError

For all 18 biomarkers in the specified order.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import torch


# Biomarker names in the EXACT order you specified
BIOMARKER_ORDER = [
    'LAV max (mL)',
    'LAV min (mL)',
    'LASV (mL)',
    'LAEF (%)',
    'RAV max (mL)',
    'RAV min (mL)',
    'RASV (mL)',
    'RAEF (%)',
    'LVEDV (mL)',
    'LVESV (mL)',
    'LVSV (mL)',
    'LVEF (%)',
    'LVCO (L/min)',
    'LVM (g)',
    'RVEDV (mL)',
    'RVESV (mL)',
    'RVSV (mL)',
    'RVEF (%)'
]


def save_predictions_to_csv(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    patient_ids: List[int],
    biomarker_names: List[str],
    output_path: str,
    reorder_to_standard: bool = True
):
    """
    Save predictions and ground truth to CSV in the standard format.
    
    Args:
        predictions: [N, 18] numpy array of predictions
        ground_truth: [N, 18] numpy array of ground truth
        patient_ids: List of N patient IDs
        biomarker_names: List of biomarker names (in model output order)
        output_path: Path to save CSV file
        reorder_to_standard: If True, reorder biomarkers to BIOMARKER_ORDER
    """
    assert predictions.shape == ground_truth.shape, "Shape mismatch!"
    assert predictions.shape[0] == len(patient_ids), "Patient ID count mismatch!"
    assert predictions.shape[1] == len(biomarker_names), "Biomarker count mismatch!"
    
    # Create mapping from model output order to standard order
    if reorder_to_standard:
        # Find indices to reorder
        reorder_indices = []
        for standard_name in BIOMARKER_ORDER:
            try:
                idx = biomarker_names.index(standard_name)
                reorder_indices.append(idx)
            except ValueError:
                raise ValueError(f"Biomarker '{standard_name}' not found in model output!")
        
        # Reorder predictions and ground truth
        predictions = predictions[:, reorder_indices]
        ground_truth = ground_truth[:, reorder_indices]
        biomarker_names_ordered = BIOMARKER_ORDER
    else:
        biomarker_names_ordered = biomarker_names
    
    # Calculate absolute errors
    abs_errors = np.abs(predictions - ground_truth)
    
    # Build DataFrame
    data = {'Patient_ID': patient_ids}
    
    # Add columns for each biomarker
    for i, biomarker in enumerate(biomarker_names_ordered):
        data[f'{biomarker}_Predicted'] = predictions[:, i]
        data[f'{biomarker}_GroundTruth'] = ground_truth[:, i]
        data[f'{biomarker}_AbsError'] = abs_errors[:, i]
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved predictions to {output_path}")
    print(f"  - {len(df)} patients")
    print(f"  - {len(biomarker_names_ordered)} biomarkers")
    
    return df


def save_predictions_to_excel(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    patient_ids: List[int],
    biomarker_names: List[str],
    output_path: str,
    reorder_to_standard: bool = True
):
    """
    Save predictions and ground truth to Excel in the standard format.
    Includes summary statistics sheet.
    
    Args:
        predictions: [N, 18] numpy array of predictions
        ground_truth: [N, 18] numpy array of ground truth
        patient_ids: List of N patient IDs
        biomarker_names: List of biomarker names (in model output order)
        output_path: Path to save Excel file
        reorder_to_standard: If True, reorder biomarkers to BIOMARKER_ORDER
    """
    # Get main dataframe
    df = save_predictions_to_csv(
        predictions, ground_truth, patient_ids, biomarker_names,
        output_path.replace('.xlsx', '_temp.csv'),  # Temp CSV
        reorder_to_standard
    )
    
    # Calculate summary statistics
    if reorder_to_standard:
        biomarker_names_ordered = BIOMARKER_ORDER
    else:
        biomarker_names_ordered = biomarker_names
    
    summary_data = []
    for biomarker in biomarker_names_ordered:
        pred_col = f'{biomarker}_Predicted'
        gt_col = f'{biomarker}_GroundTruth'
        err_col = f'{biomarker}_AbsError'
        
        # Calculate metrics
        mae = df[err_col].mean()
        rmse = np.sqrt((df[err_col] ** 2).mean())
        
        # R² score
        ss_res = ((df[gt_col] - df[pred_col]) ** 2).sum()
        ss_tot = ((df[gt_col] - df[gt_col].mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        summary_data.append({
            'Biomarker': biomarker,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Mean_GT': df[gt_col].mean(),
            'Std_GT': df[gt_col].std(),
            'Mean_Pred': df[pred_col].mean(),
            'Std_Pred': df[pred_col].std()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to Excel with multiple sheets
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Predictions', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✓ Saved predictions to Excel: {output_path}")
    print(f"  - Sheet 1: Predictions ({len(df)} patients)")
    print(f"  - Sheet 2: Summary statistics")
    
    # Remove temp CSV
    Path(output_path.parent / f"{output_path.stem}_temp.csv").unlink(missing_ok=True)
    
    return df, summary_df


def aggregate_predictions_from_batches(
    predictions_list: List[torch.Tensor],
    ground_truth_list: List[torch.Tensor],
    patient_ids_list: List[List[int]]
) -> tuple:
    """
    Aggregate predictions collected across batches.
    
    Args:
        predictions_list: List of [batch, 18] tensors
        ground_truth_list: List of [batch, 18] tensors
        patient_ids_list: List of patient ID lists
    
    Returns:
        predictions: [N, 18] numpy array
        ground_truth: [N, 18] numpy array
        patient_ids: List of N patient IDs
    """
    # Concatenate all batches
    predictions = torch.cat(predictions_list, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truth_list, dim=0).cpu().numpy()
    patient_ids = [pid for batch_ids in patient_ids_list for pid in batch_ids]
    
    return predictions, ground_truth, patient_ids


# Example usage
if __name__ == "__main__":
    print("Testing prediction saving utilities...")
    
    # Simulate some predictions
    n_samples = 100
    n_biomarkers = 18
    
    predictions = np.random.randn(n_samples, n_biomarkers) * 10 + 50
    ground_truth = predictions + np.random.randn(n_samples, n_biomarkers) * 5
    patient_ids = list(range(1000, 1000 + n_samples))
    
    # Biomarker names (in model output order - same as BIOMARKER_ORDER for this test)
    biomarker_names = BIOMARKER_ORDER.copy()
    
    # Test CSV saving
    print("\n1. Testing CSV export...")
    save_predictions_to_csv(
        predictions=predictions,
        ground_truth=ground_truth,
        patient_ids=patient_ids,
        biomarker_names=biomarker_names,
        output_path="/home/claude/test_predictions.csv"
    )
    
    # Test Excel saving
    print("\n2. Testing Excel export...")
    df, summary_df = save_predictions_to_excel(
        predictions=predictions,
        ground_truth=ground_truth,
        patient_ids=patient_ids,
        biomarker_names=biomarker_names,
        output_path="/home/claude/test_predictions.xlsx"
    )
    
    print("\n3. Summary statistics:")
    print(summary_df.head())
    
    print("\n✅ All tests passed!")