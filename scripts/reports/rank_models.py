
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
    
    # Promotion Logic
    promote_best_model(model_dir.parent / best_model['Directory'], best_model)
    
    return df

def promote_best_model(source_dir, model_info):
    """Promotes the best model to a central registry."""
    import shutil
    
    registry_path = Path(source_dir).parent.parent.parent / "models" / "registry"
    # Create class-specific subfolder (e.g., registry/two_classes_3000ep/)
    target_tag = source_dir.parent.name
    target_dir = registry_path / target_tag
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[PROMOTION] Copying best model to {target_dir}...")
    
    # Files to promote
    files_to_copy = [
        ("models/netG_final.pt", "best_generator.pt"),
        ("models/netG_final.pth", "best_generator.pth"),
        ("config.json", "config.json"),
        ("quality_report/quality_summary.json", "quality_summary.json")
    ]
    
    for src_rel, dest_name in files_to_copy:
        src_path = source_dir / src_rel
        if src_path.exists():
            shutil.copy2(src_path, target_dir / dest_name)
            print(f"  ✓ {dest_name}")

    # Save a small metadata file about the promotion
    with open(target_dir / "promotion_info.json", "w") as f:
        json.dump({
            "source_directory": source_dir.name,
            "metrics": model_info.to_dict(),
            "date_promoted": str(pd.Timestamp.now())
        }, f, indent=2)
    
    print(f"SUCCESS: Best model for {target_tag} promoted.")

if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "results/sigmoid_vs_linear"
    rank_models(base_dir)
