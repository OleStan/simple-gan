
import os
import sys
import subprocess
from pathlib import Path

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=False)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_marathon_reports.py <result_tag>")
        return
    
    tag = sys.argv[1]
    root = Path("/workspace/GANs-for-1D-Signal")
    results_base = root / "results" / tag
    
    if not results_base.exists():
        print(f"Directory not found: {results_base}")
        return

    # Improved WGAN v2
    for model_dir in results_base.glob("improved_wgan_v2_nz*"):
        if (model_dir / "models" / "netG_final.pt").exists():
            print(f"\n>>> Reporting: {model_dir.name}")
            run_cmd(f"python {root}/scripts/reports/generate_improved_wgan_v2_report.py {model_dir}")
            run_cmd(f"python {root}/scripts/reports/run_quality_check.py --model improved_wgan_v2 --model_dir {model_dir} --training_data {root}/data/training --n_generated 500 --output_dir {model_dir}/quality_report")

    # Dual WGAN
    for model_dir in results_base.glob("dual_wgan_nz*"):
        if (model_dir / "models" / "netG_final.pth").exists():
            print(f"\n>>> Reporting: {model_dir.name}")
            run_cmd(f"python {root}/scripts/reports/generate_dual_wgan_report.py {model_dir}")
            run_cmd(f"python {root}/scripts/reports/run_quality_check.py --model dual_wgan --model_dir {model_dir} --training_data {root}/data/training --n_generated 500 --output_dir {model_dir}/quality_report")

    # NEW: Automatically rank and promote the winner for this tag
    print(f"\n>>> Ranking and Promoting Winner for {tag}...")
    from scripts.reports.rank_models import rank_models
    rank_models(results_base)

if __name__ == "__main__":
    main()
