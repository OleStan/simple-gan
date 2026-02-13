import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from preprocessing_mddect import Dataset


def _sorted_epoch_images(img_dir: str) -> list[str]:
    paths = []
    for name in os.listdir(img_dir):
        if not (name.startswith('wgan_mddect_epoch_') and name.endswith('.png')):
            continue
        paths.append(os.path.join(img_dir, name))

    def _epoch_num(path: str) -> int:
        base = os.path.basename(path)
        num_part = base.replace('wgan_mddect_epoch_', '').replace('.png', '')
        try:
            return int(num_part)
        except ValueError:
            return 10**9

    return sorted(paths, key=_epoch_num)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate report for MDDECT WGAN run')
    parser.add_argument('--data-root', type=str, required=True, help='Folder with extracted .txt signals (e.g. data/mddect_normal_ch0)')
    parser.add_argument('--nets-dir', type=str, required=True, help='Folder containing wgan_mddect_netG.pkl and wgan_mddect_netD.pkl')
    parser.add_argument('--img-dir', type=str, required=True, help='Folder with epoch images wgan_mddect_epoch_*.png')
    parser.add_argument('--out-root', type=str, default='./results_mddect', help='Root folder where a timestamped report folder is created')
    parser.add_argument('--signal-length', type=int, default=1024)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num-compare', type=int, default=10)
    parser.add_argument('--num-generated', type=int, default=100)
    parser.add_argument('--gif-ms', type=int, default=200)
    args = parser.parse_args()

    missing = []
    if not os.path.isdir(args.data_root):
        missing.append(f"data-root: {args.data_root}")
    if not os.path.isdir(args.nets_dir):
        missing.append(f"nets-dir: {args.nets_dir}")
    if not os.path.isdir(args.img_dir):
        missing.append(f"img-dir: {args.img_dir}")

    if missing:
        hint = (
            "One or more required folders do not exist.\n\n"
            "Hint: `--data-root` must point to the folder with extracted *.txt signals. "
            "If you trained a *_v2 run (different nets/img output dirs), you typically still reuse the same extracted data folder "
            "(e.g. `data/mddect_1p9mm_angle3_ch0` or `data/mddect_train_1p9mm_angle3_ch0`)."
        )
        raise FileNotFoundError(hint + "\n\nMissing:\n- " + "\n- ".join(missing))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.out_root, exist_ok=True)
    run_dir = os.path.join(args.out_root, 'mddect_wgan_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    viz_dir = os.path.join(run_dir, 'visualizations')
    models_dir = os.path.join(run_dir, 'models')
    train_images_dir = os.path.join(run_dir, 'training_images')
    gen_samples_dir = os.path.join(run_dir, 'generated_samples')

    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(gen_samples_dir, exist_ok=True)

    dataset = Dataset(args.data_root, target_length=args.signal_length)

    netG_path = os.path.join(args.nets_dir, 'wgan_mddect_netG.pkl')
    netD_path = os.path.join(args.nets_dir, 'wgan_mddect_netD.pkl')

    netG = torch.load(netG_path, map_location=device, weights_only=False)
    netG.eval()

    noise = torch.randn(args.num_generated, args.nz, 1, device=device)
    with torch.no_grad():
        generated = netG(noise).detach().cpu().numpy().reshape(args.num_generated, -1)

    for i in range(args.num_generated):
        np.savetxt(os.path.join(gen_samples_dir, f'generated_signal_{i:03d}.txt'), generated[i], fmt='%.6f')

    # Comparison plot (real vs generated)
    compare_n = min(args.num_compare, max(1, len(dataset)))
    rows = (compare_n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 4 * rows))
    if rows == 1:
        axes = np.array([axes])

    fig.suptitle('MDDECT WGAN: Real vs Generated (random pairs)', fontsize=16, fontweight='bold')

    for idx in range(compare_n):
        ax = axes[idx // 2, idx % 2]
        real_signal, _ = dataset[np.random.randint(0, len(dataset))]
        real_signal = real_signal.numpy().flatten()
        fake_signal = generated[np.random.randint(0, args.num_generated)]

        x = np.arange(len(real_signal))
        ax.plot(x, real_signal, 'b-', alpha=0.7, linewidth=1.5, label='Real')
        ax.plot(x, fake_signal, 'r-', alpha=0.7, linewidth=1.5, label='Generated')
        ax.set_title(f'Pair {idx + 1}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Normalized Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    # Hide any unused subplot (when compare_n is odd)
    if compare_n % 2 == 1:
        axes[-1, -1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'comparison_real_vs_generated.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Statistical comparison
    sample_count = min(50, len(dataset))
    real_samples = []
    for _ in range(sample_count):
        sig, _ = dataset[np.random.randint(0, len(dataset))]
        real_samples.append(sig.numpy().flatten())
    real_samples = np.array(real_samples)

    fake_samples = generated[:sample_count]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MDDECT WGAN: Statistical Analysis', fontsize=16, fontweight='bold')

    axes[0, 0].plot(real_samples.mean(axis=0), 'b-', linewidth=2, label='Real Mean')
    axes[0, 0].fill_between(
        range(real_samples.shape[1]),
        real_samples.mean(axis=0) - real_samples.std(axis=0),
        real_samples.mean(axis=0) + real_samples.std(axis=0),
        alpha=0.3,
        color='blue',
    )
    axes[0, 0].set_title('Real: Mean ± Std', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(fake_samples.mean(axis=0), 'r-', linewidth=2, label='Generated Mean')
    axes[0, 1].fill_between(
        range(fake_samples.shape[1]),
        fake_samples.mean(axis=0) - fake_samples.std(axis=0),
        fake_samples.mean(axis=0) + fake_samples.std(axis=0),
        alpha=0.3,
        color='red',
    )
    axes[0, 1].set_title('Generated: Mean ± Std', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(real_samples.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_title('Real Value Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(fake_samples.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_title('Generated Value Distribution', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'statistical_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Summary plot
    real_mean = real_samples.mean(axis=0)
    fake_mean = fake_samples.mean(axis=0)
    mse = float(np.mean((real_mean - fake_mean) ** 2))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(real_mean, 'b-', linewidth=2)
    ax1.set_title('Real Mean', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fake_mean, 'r-', linewidth=2)
    ax2.set_title('Generated Mean', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(real_mean, 'b-', linewidth=2, label='Real Mean')
    ax3.plot(fake_mean, 'r--', linewidth=2, label='Generated Mean')
    ax3.set_title('Mean Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(real_mean - fake_mean, 'g-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_title(f'Difference (MSE: {mse:.6f})', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('MDDECT WGAN Report Summary', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(run_dir, 'SUMMARY.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # GIF + training images
    epoch_paths = _sorted_epoch_images(args.img_dir)
    pil_images = [Image.open(p) for p in epoch_paths]

    if pil_images:
        pil_images[0].save(
            os.path.join(viz_dir, 'training_progression.gif'),
            save_all=True,
            append_images=pil_images[1:],
            duration=args.gif_ms,
            loop=0,
        )

    for i, p in enumerate(epoch_paths):
        shutil.copy2(p, os.path.join(train_images_dir, f'epoch_{i:03d}.png'))

    # Copy models
    shutil.copy2(netG_path, os.path.join(models_dir, 'generator.pkl'))
    if os.path.exists(netD_path):
        shutil.copy2(netD_path, os.path.join(models_dir, 'discriminator.pkl'))

    # README
    readme_path = os.path.join(run_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('# MDDECT WGAN Report\n')
        f.write(f'Generated: {datetime.now().isoformat(timespec="seconds")}\n\n')
        f.write('## Inputs\n')
        f.write(f'- data-root: `{args.data_root}`\n')
        f.write(f'- nets-dir: `{args.nets_dir}`\n')
        f.write(f'- img-dir: `{args.img_dir}`\n\n')
        f.write('## Outputs\n')
        f.write(f'- `SUMMARY.png`\n')
        f.write(f'- `visualizations/comparison_real_vs_generated.png`\n')
        f.write(f'- `visualizations/statistical_analysis.png`\n')
        f.write(f'- `visualizations/training_progression.gif`\n')
        f.write(f'- `generated_samples/` ({args.num_generated} samples)\n')
        f.write(f'- `models/` (generator/discriminator)\n\n')
        f.write('## Metrics\n')
        f.write(f'- mean MSE (real vs generated): `{mse:.6f}`\n')

    print(f'Report generated in: {run_dir}')


if __name__ == '__main__':
    main()
