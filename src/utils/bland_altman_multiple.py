"""
Bland-Altman Analysis and Agreement Metrics for Cardiac Biomarker Predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro
import re


def slugify(name: str) -> str:
    """Convert biomarker names to consistent format."""
    name = name.replace("(%)", "pct")
    name = name.replace("%", "pct")
    name = name.replace("(mL)", "mL")
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to pred_* and target_* format."""
    rename_map = {}
    for col in df.columns:
        if col.endswith("_Pred"):
            base = col[: -len("_Pred")].strip()
            slug = slugify(base)
            rename_map[col] = f"pred_{slug}"
        elif col.endswith("_GT"):
            base = col[: -len("_GT")].strip()
            slug = slugify(base)
            rename_map[col] = f"target_{slug}"
    return df.rename(columns=rename_map)


class BlandAltmanAnalyzer:
    """Performs Bland-Altman analysis and related agreement metrics."""
    
    def __init__(self, predictions_df, biomarker_names=None):
        self.df = predictions_df
        if biomarker_names is None:
            self.biomarker_names = self._infer_biomarker_names()
        else:
            self.biomarker_names = biomarker_names
        self.results = {}
    
    def _infer_biomarker_names(self):
        pred_cols = [col for col in self.df.columns if col.startswith('pred_')]
        return [col.replace('pred_', '') for col in pred_cols]
    
    def compute_bland_altman_metrics(self, predictions, targets, biomarker_name):
        """Compute Bland-Altman metrics in both absolute and percentage domains."""
        differences = targets - predictions
        means = (targets + predictions) / 2
        
        bias = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        upper_loa = bias + 1.96 * std_diff
        lower_loa = bias - 1.96 * std_diff
        
        n = len(differences)
        se_bias = std_diff / np.sqrt(n) if n > 0 else np.nan
        se_loa = std_diff * np.sqrt(3 / n) if n > 0 else np.nan
        
        bias_ci_lower = bias - 1.96 * se_bias
        bias_ci_upper = bias + 1.96 * se_bias
        upper_loa_ci_lower = upper_loa - 1.96 * se_loa
        upper_loa_ci_upper = upper_loa + 1.96 * se_loa
        lower_loa_ci_lower = lower_loa - 1.96 * se_loa
        lower_loa_ci_upper = lower_loa + 1.96 * se_loa
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(means, differences)
        has_proportional_bias = p_value < 0.05
        
        if n >= 3:
            shapiro_stat, shapiro_p = shapiro(differences)
            differences_normal = shapiro_p > 0.05
        else:
            shapiro_stat, shapiro_p = None, None
            differences_normal = None
        
        within_loa = np.sum((differences >= lower_loa) & (differences <= upper_loa))
        pct_within_loa = 100 * within_loa / n if n > 0 else np.nan
        
        mae = np.mean(np.abs(differences))
        rmse = np.sqrt(np.mean(differences**2))
        
        # Percentage domain
        den = np.where(targets == 0, np.nan, targets)
        differences_pct = 100.0 * differences / den
        diffs_pct_valid = differences_pct[~np.isnan(differences_pct)]
        
        if diffs_pct_valid.size > 1:
            bias_pct = np.nanmean(diffs_pct_valid)
            std_diff_pct = np.nanstd(diffs_pct_valid, ddof=1)
            upper_loa_pct = bias_pct + 1.96 * std_diff_pct
            lower_loa_pct = bias_pct - 1.96 * std_diff_pct
            se_bias_pct = std_diff_pct / np.sqrt(diffs_pct_valid.size)
            se_loa_pct = std_diff_pct * np.sqrt(3 / diffs_pct_valid.size)
            bias_ci_lower_pct = bias_pct - 1.96 * se_bias_pct
            bias_ci_upper_pct = bias_pct + 1.96 * se_bias_pct
            upper_loa_ci_lower_pct = upper_loa_pct - 1.96 * se_loa_pct
            upper_loa_ci_upper_pct = upper_loa_pct + 1.96 * se_loa_pct
            lower_loa_ci_lower_pct = lower_loa_pct - 1.96 * se_loa_pct
            lower_loa_ci_upper_pct = lower_loa_pct + 1.96 * se_loa_pct
            within_loa_pct = np.sum((diffs_pct_valid >= lower_loa_pct) & (diffs_pct_valid <= upper_loa_pct))
            pct_within_loa_pct = 100.0 * within_loa_pct / diffs_pct_valid.size
            mae_pct = np.nanmean(np.abs(diffs_pct_valid))
            rmse_pct = np.sqrt(np.nanmean(diffs_pct_valid**2))
        else:
            bias_pct = std_diff_pct = upper_loa_pct = lower_loa_pct = np.nan
            bias_ci_lower_pct = bias_ci_upper_pct = np.nan
            upper_loa_ci_lower_pct = upper_loa_ci_upper_pct = np.nan
            lower_loa_ci_lower_pct = lower_loa_ci_upper_pct = np.nan
            pct_within_loa_pct = mae_pct = rmse_pct = np.nan
        
        mape = np.nanmean(np.abs(differences / den)) * 100.0
        pearson_r, pearson_p = stats.pearsonr(predictions, targets)
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        icc = self._compute_icc(predictions, targets)
        
        return {
            'biomarker': biomarker_name,
            'n_samples': n,
            'bias': bias,
            'bias_ci': (bias_ci_lower, bias_ci_upper),
            'std_diff': std_diff,
            'upper_loa': upper_loa,
            'upper_loa_ci': (upper_loa_ci_lower, upper_loa_ci_upper),
            'lower_loa': lower_loa,
            'lower_loa_ci': (lower_loa_ci_lower, lower_loa_ci_upper),
            'bias_pct': bias_pct,
            'bias_ci_pct': (bias_ci_lower_pct, bias_ci_upper_pct),
            'std_diff_pct': std_diff_pct,
            'upper_loa_pct': upper_loa_pct,
            'upper_loa_ci_pct': (upper_loa_ci_lower_pct, upper_loa_ci_upper_pct),
            'lower_loa_pct': lower_loa_pct,
            'lower_loa_ci_pct': (lower_loa_ci_lower_pct, lower_loa_ci_upper_pct),
            'pct_within_loa_pct': pct_within_loa_pct,
            'mae_pct': mae_pct,
            'rmse_pct': rmse_pct,
            'proportional_bias_slope': slope,
            'proportional_bias_pvalue': p_value,
            'has_proportional_bias': has_proportional_bias,
            'shapiro_stat': shapiro_stat,
            'shapiro_pvalue': shapiro_p,
            'differences_normal': differences_normal,
            'pct_within_loa': pct_within_loa,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'r2': r2,
            'icc': icc,
            'means': means,
            'differences': differences,
            'differences_pct': differences_pct
        }
    
    def _compute_icc_1_1(self, predictions, targets):
        data = np.column_stack([targets, predictions])
        n = len(targets)
        k = 2
        grand_mean = np.mean(data)
        subject_means = np.mean(data, axis=1)
        ms_between = k * np.sum((subject_means - grand_mean)**2) / (n - 1)
        ms_within = np.sum((data - subject_means[:, np.newaxis])**2) / (n * (k - 1))
        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        return icc
    
    def _compute_icc(self, predictions, targets):
        """
        Compute ICC(2,1): Two-way random effects, absolute agreement, single measurement.
        Based on Shrout & Fleiss (1979).
        
        Parameters
        ----------
        predictions : array-like
            Model predictions (n subjects)
        targets : array-like
            Reference/ground truth measurements (n subjects)

        Returns
        -------
        icc : float
            ICC(2,1) value
        """
        # Stack data: n subjects × k raters (here k = 2: target, prediction)
        data = np.column_stack([targets, predictions])
        n, k = data.shape  # n subjects, k raters

        # Mean per subject and per rater, and grand mean
        subject_means = np.mean(data, axis=1, keepdims=True)
        rater_means = np.mean(data, axis=0, keepdims=True)
        grand_mean = np.mean(data)

        # Two-way ANOVA mean squares
        # MS_subjects: between subjects
        ms_subjects = k * np.sum((subject_means - grand_mean) ** 2) / (n - 1)

        # MS_raters: between raters
        ms_raters = n * np.sum((rater_means - grand_mean) ** 2) / (k - 1)

        # MS_error: residual
        ms_error = np.sum((data - subject_means - rater_means + grand_mean) ** 2) / ((n - 1) * (k - 1))

        # ICC(2,1) formula
        icc = (ms_subjects - ms_error) / (ms_subjects + (k - 1) * ms_error + (k * (ms_raters - ms_error) / n))
        return icc
    
    def plot_bland_altman(self, metrics, output_path, figsize=(10, 7)):
        means = metrics['means']
        differences = metrics['differences']
        bias = metrics['bias']
        upper_loa = metrics['upper_loa']
        lower_loa = metrics['lower_loa']
        biomarker = metrics['biomarker']
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(means, differences, alpha=0.55, s=28, edgecolors='k', linewidth=0.4, zorder=2)
        ax.axhline(bias, color='blue', linestyle='--', linewidth=2, label=f'Bias: {bias:.3f}', zorder=1)
        ax.axhline(upper_loa, color='red', linestyle='--', linewidth=2, label=f'Upper LoA: {upper_loa:.3f}', zorder=1)
        ax.axhline(lower_loa, color='red', linestyle='--', linewidth=2, label=f'Lower LoA: {lower_loa:.3f}', zorder=1)
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=1)
        
        bias_ci = metrics['bias_ci']
        upper_loa_ci = metrics['upper_loa_ci']
        lower_loa_ci = metrics['lower_loa_ci']
        x_min, x_max = ax.get_xlim()
        ax.fill_between([x_min, x_max], bias_ci[0], bias_ci[1], color='blue', alpha=0.10, zorder=0)
        ax.fill_between([x_min, x_max], upper_loa_ci[0], upper_loa_ci[1], color='red', alpha=0.10, zorder=0)
        ax.fill_between([x_min, x_max], lower_loa_ci[0], lower_loa_ci[1], color='red', alpha=0.10, zorder=0)
        
        if metrics['has_proportional_bias']:
            slope = metrics['proportional_bias_slope']
            x_range = np.array([means.min(), means.max()])
            y_range = slope * (x_range - means.mean()) + bias
            ax.plot(x_range, y_range, 'g--', linewidth=2, alpha=0.7,
                    label=f'Proportional bias (p={metrics["proportional_bias_pvalue"]:.4f})', zorder=1)
        
        ax.set_xlabel('Mean of Ground Truth and Prediction', fontsize=12)
        ax.set_ylabel('Difference (Ground Truth - Prediction)', fontsize=12)
        ax.set_title(f'Bland-Altman Plot: {biomarker}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        stats_label = '\n'.join([
            f'n = {metrics["n_samples"]}',
            f'Within LoA: {metrics["pct_within_loa"]:.1f}%',
            f'MAE: {metrics["mae"]:.3f}',
            f'RMSE: {metrics["rmse"]:.3f}',
        ])
        from matplotlib.lines import Line2D
        stats_handle = Line2D([], [], linestyle='none', marker='', label=stats_label)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        final_labels = [stats_label] + list(by_label.keys())
        final_handles = [stats_handle] + [by_label[k] for k in by_label.keys()]
        
        ax.legend(final_handles, final_labels, loc='upper left', bbox_to_anchor=(1.02, 1),
                borderaxespad=0., frameon=True, fancybox=True, handlelength=0, 
                handletextpad=0.4, labelspacing=0.6)
        
        plt.tight_layout(rect=[0, 0, 0.78, 1])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_bland_altman_percent(self, metrics, output_path, figsize=(10, 7)):
        means = metrics['means']
        differences_pct = metrics['differences_pct']
        bias = metrics['bias_pct']
        upper_loa = metrics['upper_loa_pct']
        lower_loa = metrics['lower_loa_pct']
        biomarker = metrics['biomarker']
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(means, differences_pct, alpha=0.55, s=28, edgecolors='k', linewidth=0.4, zorder=2)
        ax.axhline(bias, color='blue', linestyle='--', linewidth=2, label=f'Bias: {bias:.2f}%', zorder=1)
        ax.axhline(upper_loa, color='red', linestyle='--', linewidth=2, label=f'Upper LoA: {upper_loa:.2f}%', zorder=1)
        ax.axhline(lower_loa, color='red', linestyle='--', linewidth=2, label=f'Lower LoA: {lower_loa:.2f}%', zorder=1)
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=1)
        
        bias_ci = metrics['bias_ci_pct']
        upper_loa_ci = metrics['upper_loa_ci_pct']
        lower_loa_ci = metrics['lower_loa_ci_pct']
        x_min, x_max = ax.get_xlim()
        ax.fill_between([x_min, x_max], bias_ci[0], bias_ci[1], color='blue', alpha=0.10, zorder=0)
        ax.fill_between([x_min, x_max], upper_loa_ci[0], upper_loa_ci[1], color='red', alpha=0.10, zorder=0)
        ax.fill_between([x_min, x_max], lower_loa_ci[0], lower_loa_ci[1], color='red', alpha=0.10, zorder=0)
        
        ax.set_xlabel('Mean of Ground Truth and Prediction (native units)', fontsize=12)
        ax.set_ylabel('Relative Difference (%): 100 × (GT − Pred) / GT', fontsize=12)
        ax.set_title(f'Bland-Altman Plot (%): {biomarker}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        stats_label = '\n'.join([
            f'n = {metrics["n_samples"]}',
            f'Within LoA: {metrics["pct_within_loa_pct"]:.1f}%',
            f'MAE%: {metrics["mae_pct"]:.2f}',
            f'RMSPE%: {metrics["rmse_pct"]:.2f}',
        ])
        from matplotlib.lines import Line2D
        stats_handle = Line2D([], [], linestyle='none', marker='', label=stats_label)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        final_labels = [stats_label] + list(by_label.keys())
        final_handles = [stats_handle] + [by_label[k] for k in by_label.keys()]
        
        ax.legend(final_handles, final_labels, loc='upper left', bbox_to_anchor=(1.02, 1),
                borderaxespad=0., frameon=True, fancybox=True, handlelength=0,
                handletextpad=0.4, labelspacing=0.6)
        
        plt.tight_layout(rect=[0, 0, 0.78, 1])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_scatter_with_identity(self, predictions, targets, biomarker, output_path, figsize=(8, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(targets, predictions, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2,
                label='Perfect agreement', alpha=0.7)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(targets, predictions)
        x_range = np.array([min_val, max_val])
        y_range = slope * x_range + intercept
        ax.plot(x_range, y_range, 'r-', linewidth=2, label=f'Regression (R²={r_value**2:.3f})')
        
        ax.set_xlabel('Ground Truth', fontsize=12)
        ax.set_ylabel('Prediction', fontsize=12)
        ax.set_title(f'Prediction vs Ground Truth: {biomarker}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        mae = np.mean(np.abs(targets - predictions))
        rmse = np.sqrt(np.mean((targets - predictions)**2))
        textstr = '\n'.join([
            f'MAE: {mae:.3f}',
            f'RMSE: {rmse:.3f}',
            f'R²: {r_value**2:.3f}',
            f'y = {slope:.3f}x + {intercept:.3f}'
        ])
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_residuals(self, predictions, targets, biomarker, output_path, figsize=(10, 6)):
        residuals = targets - predictions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        ax1.scatter(predictions, residuals, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        ax1.axhline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values', fontsize=11)
        ax1.set_ylabel('Residuals (Ground Truth - Predicted)', fontsize=11)
        ax1.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{biomarker} - Residual Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_all_biomarkers(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        (output_path / 'bland_altman').mkdir(exist_ok=True)
        (output_path / 'scatter').mkdir(exist_ok=True)
        (output_path / 'residuals').mkdir(exist_ok=True)
        
        all_metrics = []
        
        for biomarker in self.biomarker_names:
            print(f"\nAnalyzing {biomarker}...")
            
            pred_col = f'pred_{biomarker}'
            target_col = f'target_{biomarker}'
            
            if pred_col not in self.df.columns or target_col not in self.df.columns:
                print(f"  Skipping {biomarker}: columns not found")
                continue
            
            predictions = self.df[pred_col].values
            targets = self.df[target_col].values
            
            mask = ~(np.isnan(predictions) | np.isnan(targets))
            predictions = predictions[mask]
            targets = targets[mask]
            
            if len(predictions) == 0:
                print(f"  Skipping {biomarker}: no valid data")
                continue
            
            metrics = self.compute_bland_altman_metrics(predictions, targets, biomarker)
            all_metrics.append(metrics)
            
            self.plot_bland_altman(metrics, output_path / 'bland_altman' / f'{biomarker}_bland_altman.png')
            self.plot_bland_altman_percent(metrics, output_path / 'bland_altman' / f'{biomarker}_bland_altman_pct.png')
            self.plot_scatter_with_identity(predictions, targets, biomarker, output_path / 'scatter' / f'{biomarker}_scatter.png')
            self.plot_residuals(predictions, targets, biomarker, output_path / 'residuals' / f'{biomarker}_residuals.png')
            
            print(f"  ✓ Completed analysis for {biomarker}")
        
        self._save_summary_metrics(all_metrics, output_path)
        
        print(f"\n✓ Analysis complete! Results saved to {output_path}")
        return all_metrics
    
    def _save_summary_metrics(self, all_metrics, output_path):
        summary_rows = []
        for m in all_metrics:
            summary_rows.append({
                'Biomarker': m['biomarker'],
                'N': m['n_samples'],
                'Bias': m['bias'],
                'Bias_CI_Lower': m['bias_ci'][0],
                'Bias_CI_Upper': m['bias_ci'][1],
                'Std_Diff': m['std_diff'],
                'Upper_LoA': m['upper_loa'],
                'Lower_LoA': m['lower_loa'],
                'Pct_Within_LoA_abs': m['pct_within_loa'],
                'MAE': m['mae'],
                'RMSE': m['rmse'],
                'Bias_%': m['bias_pct'],
                'Bias_%_CI_Lower': m['bias_ci_pct'][0],
                'Bias_%_CI_Upper': m['bias_ci_pct'][1],
                'Std_Diff_%': m['std_diff_pct'],
                'Upper_LoA_%': m['upper_loa_pct'],
                'Lower_LoA_%': m['lower_loa_pct'],
                'Pct_Within_LoA_%': m['pct_within_loa_pct'],
                'MAE_%': m['mae_pct'],
                'RMSPE_%': m['rmse_pct'],
                'MAPE': m['mape'],
                'Pearson_R': m['pearson_r'],
                'R²': m['r2'],
                'ICC': m['icc'],
                'Proportional_Bias': m['has_proportional_bias'],
                'Proportional_Bias_P': m['proportional_bias_pvalue'],
                'Differences_Normal': m['differences_normal'],
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_path / 'summary_metrics.csv', index=False)
        print(f"\n✓ Summary metrics saved: {output_path / 'summary_metrics.csv'}")
        #save xlsx
        summary_df.to_excel(output_path / 'summary_metrics.xlsx', index=False)
        print(f"\n✓ Summary metrics saved: {output_path / 'summary_metrics.xlsx'}")


            

def main_single_run(predictions_file, output_dir, biomarkers=None):
    """Analyze a single run."""
    print(f"Loading predictions from {predictions_file}...")
    df = pd.read_csv(predictions_file)
    df = standardize_columns(df)
    print(f"Loaded {len(df)} predictions")
    
    analyzer = BlandAltmanAnalyzer(df, biomarker_names=biomarkers)
    print(f"\nBiomarkers to analyze: {', '.join(analyzer.biomarker_names)}")
    
    metrics = analyzer.analyze_all_biomarkers(output_dir)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    for m in metrics:
        print(f"\n{m['biomarker']}:")
        print(f"  Bias: {m['bias']:.4f} (95% CI: [{m['bias_ci'][0]:.4f}, {m['bias_ci'][1]:.4f}])")
        print(f"  LoA: [{m['lower_loa']:.4f}, {m['upper_loa']:.4f}]")
        print(f"  Within LoA: {m['pct_within_loa']:.1f}%")
        print(f"  MAE: {m['mae']:.4f}, RMSE: {m['rmse']:.4f}")
        print(f"  R²: {m['r2']:.4f}, ICC: {m['icc']:.4f}")
