
import json
import os
from pathlib import Path
import pandas as pd

def rank_models(results_base_dir):
    results_path = Path(results_base_dir)
    data = []

    # Find all quality summaries
    summary_files = list(results_path.glob("**/quality_summary.json"))
    
    if not summary_files:
        print(f"No quality_summary.json files found in {results_base_dir}")
        return

    for file_path in summary_files:
        with open(file_path, 'r') as f:
            summary = json.load(f)
        
        # Extract model info from path or summary
        model_dir = file_path.parent.parent
        model_name = summary.get('model_name', 'Unknown')
        
        # Load config to get nz
        nz = "N/A"
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                nz = cfg.get('nz', 'N/A')

        metrics = summary.get('criteria', {})
        
        # Extract individual metrics
        mmd = metrics.get('2_distribution_distance', {}).get('mmd_score', 1.0)
        wasserstein = metrics.get('2_distribution_distance', {}).get('wasserstein_mean', 1.0)
        lipschitz = metrics.get('5_noise_robustness', {}).get('mean_lipschitz', 5.0)
        moment_diff = metrics.get('1_moment_matching', {}).get('mean_rel_diff', 1.0)
        
        # Normalize metrics (lower is better for all of these)
        # Note: In a real "Auto-Tuning" scenario, we'd normalize these against the min/max of the whole batch
        
        # Calculate Composite Score (Lower is Better)
        # Weighting: 40% MMD, 30% Wasserstein, 20% Lipschitz, 10% Moments
        score = (mmd * 0.4) + (wasserstein * 0.3) + (lipschitz * 0.2 * 0.1) + (moment_diff * 0.1)
        
        data.append({
            'NZ': nz,
            'Model': model_name,
            'Score': round(score, 4),
            'MMD': round(mmd, 4),
            'W-Dist': round(wasserstein, 4),
            'Lipschitz': round(lipschitz, 2),
            'Moments': round(moment_diff, 4),
            'Directory': model_dir.name
        })

    df = pd.DataFrame(data)
    if df.empty:
        return

    # Sort by score ascending
    df = df.sort_values(by='Score')
    
    print("\n" + "="*90)
    print("  AUTOMATED HYPERPARAMETER TUNING: MODEL RANKING")
    print("  (Ranked by Composite Quality Score - Lower is Better)")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)
    
    best_model = df.iloc[0]
    print(f"\nWINNER: {best_model['Model']} with nz={best_model['NZ']}")
    print(f"Path: {best_model['Directory']}")
    
    return df

if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "results/sigmoid_vs_linear"
    rank_models(base_dir)
