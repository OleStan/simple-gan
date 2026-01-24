import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
from datetime import datetime
from preprocessing import Dataset
from wgan import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

results_root = './results'
os.makedirs(results_root, exist_ok=True)

output_dir = os.path.join(results_root, 'wgan_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'training_images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'generated_samples'), exist_ok=True)

print("Loading dataset...")
dataset = Dataset('./data/brilliant_blue')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

print("Loading trained generator...")
nz = 100
netG = torch.load('./nets/wgan_netG.pkl', map_location=device, weights_only=False)
netG.eval()

print("\n1. Generating synthetic signals...")
num_samples = 10
noise = torch.randn(num_samples, nz, 1, device=device)
with torch.no_grad():
    fake_signals = netG(noise).cpu().numpy()

print("2. Creating comparison plots...")
fig, axes = plt.subplots(5, 2, figsize=(15, 20))
fig.suptitle('Generated vs Real Raman Spectra Comparison', fontsize=16, fontweight='bold')

for i in range(num_samples):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    real_signal, _ = dataset[np.random.randint(0, len(dataset))]
    real_signal = real_signal.numpy().flatten()
    fake_signal = fake_signals[i].flatten()
    
    x_axis = np.arange(len(real_signal))
    
    ax.plot(x_axis, real_signal, 'b-', alpha=0.7, linewidth=1.5, label='Real Signal')
    ax.plot(x_axis, fake_signal, 'r-', alpha=0.7, linewidth=1.5, label='Generated Signal')
    ax.set_xlabel('Wavenumber Index', fontsize=10)
    ax.set_ylabel('Normalized Intensity', fontsize=10)
    ax.set_title(f'Sample {i+1}', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'visualizations', 'comparison_real_vs_generated.png'), dpi=300, bbox_inches='tight')
print(f"   Saved: comparison_real_vs_generated.png")
plt.close()

print("3. Creating statistical comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Statistical Analysis: Real vs Generated Signals', fontsize=16, fontweight='bold')

real_samples = []
for i in range(50):
    sig, _ = dataset[np.random.randint(0, len(dataset))]
    real_samples.append(sig.numpy().flatten())
real_samples = np.array(real_samples)

noise = torch.randn(50, nz, 1, device=device)
with torch.no_grad():
    fake_samples = netG(noise).cpu().numpy().reshape(50, -1)

axes[0, 0].plot(real_samples.mean(axis=0), 'b-', linewidth=2, label='Real Mean')
axes[0, 0].fill_between(range(real_samples.shape[1]), 
                         real_samples.mean(axis=0) - real_samples.std(axis=0),
                         real_samples.mean(axis=0) + real_samples.std(axis=0),
                         alpha=0.3, color='blue')
axes[0, 0].set_title('Real Signals: Mean ± Std', fontweight='bold')
axes[0, 0].set_xlabel('Wavenumber Index')
axes[0, 0].set_ylabel('Normalized Intensity')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(fake_samples.mean(axis=0), 'r-', linewidth=2, label='Generated Mean')
axes[0, 1].fill_between(range(fake_samples.shape[1]), 
                         fake_samples.mean(axis=0) - fake_samples.std(axis=0),
                         fake_samples.mean(axis=0) + fake_samples.std(axis=0),
                         alpha=0.3, color='red')
axes[0, 1].set_title('Generated Signals: Mean ± Std', fontweight='bold')
axes[0, 1].set_xlabel('Wavenumber Index')
axes[0, 1].set_ylabel('Normalized Intensity')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

axes[1, 0].hist(real_samples.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black', label='Real')
axes[1, 0].set_title('Distribution of Intensity Values', fontweight='bold')
axes[1, 0].set_xlabel('Normalized Intensity')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(fake_samples.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black', label='Generated')
axes[1, 1].set_title('Distribution of Intensity Values', fontweight='bold')
axes[1, 1].set_xlabel('Normalized Intensity')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'visualizations', 'statistical_analysis.png'), dpi=300, bbox_inches='tight')
print(f"   Saved: statistical_analysis.png")
plt.close()

print("4. Creating training progression GIF...")
epoch_images = []
for epoch in range(64):
    img_path = f'./img/wgan_epoch_{epoch}.png'
    if os.path.exists(img_path):
        epoch_images.append(Image.open(img_path))

if epoch_images:
    gif_path = os.path.join(output_dir, 'visualizations', 'training_progression.gif')
    epoch_images[0].save(
        gif_path,
        save_all=True,
        append_images=epoch_images[1:],
        duration=200,
        loop=0
    )
    print(f"   Saved: training_progression.gif ({len(epoch_images)} frames)")

print("5. Copying training images...")
for epoch in range(64):
    src = f'./img/wgan_epoch_{epoch}.png'
    if os.path.exists(src):
        dst = os.path.join(output_dir, 'training_images', f'epoch_{epoch:03d}.png')
        shutil.copy2(src, dst)
print(f"   Copied {64} training images")

print("6. Copying trained models...")
shutil.copy2('./nets/wgan_netG.pkl', os.path.join(output_dir, 'models', 'generator.pkl'))
shutil.copy2('./nets/wgan_netD.pkl', os.path.join(output_dir, 'models', 'discriminator.pkl'))
print("   Saved: generator.pkl, discriminator.pkl")

print("7. Generating sample outputs...")
num_generated = 100
noise = torch.randn(num_generated, nz, 1, device=device)
with torch.no_grad():
    generated_signals = netG(noise).cpu().numpy()

for i in range(num_generated):
    signal = generated_signals[i].flatten()
    np.savetxt(os.path.join(output_dir, 'generated_samples', f'generated_signal_{i:03d}.txt'), signal, fmt='%.6f')
print(f"   Saved {num_generated} generated signals as txt files")

print("8. Creating final summary plot...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
for i in range(min(5, num_generated)):
    ax1.plot(generated_signals[i].flatten(), alpha=0.6, linewidth=1.5, label=f'Sample {i+1}')
ax1.set_title('Generated Raman Spectra Samples', fontsize=14, fontweight='bold')
ax1.set_xlabel('Wavenumber Index', fontsize=11)
ax1.set_ylabel('Normalized Intensity', fontsize=11)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
real_signal, _ = dataset[0]
ax2.plot(real_signal.numpy().flatten(), 'b-', linewidth=2)
ax2.set_title('Example Real Signal', fontsize=12, fontweight='bold')
ax2.set_xlabel('Wavenumber Index', fontsize=10)
ax2.set_ylabel('Normalized Intensity', fontsize=10)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(generated_signals[0].flatten(), 'r-', linewidth=2)
ax3.set_title('Example Generated Signal', fontsize=12, fontweight='bold')
ax3.set_xlabel('Wavenumber Index', fontsize=10)
ax3.set_ylabel('Normalized Intensity', fontsize=10)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[2, 0])
real_mean = real_samples.mean(axis=0)
fake_mean = fake_samples.mean(axis=0)
ax4.plot(real_mean, 'b-', linewidth=2, label='Real Mean', alpha=0.8)
ax4.plot(fake_mean, 'r--', linewidth=2, label='Generated Mean', alpha=0.8)
ax4.set_title('Mean Signal Comparison', fontsize=12, fontweight='bold')
ax4.set_xlabel('Wavenumber Index', fontsize=10)
ax4.set_ylabel('Normalized Intensity', fontsize=10)
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[2, 1])
mse = np.mean((real_mean - fake_mean) ** 2)
ax5.plot(real_mean - fake_mean, 'g-', linewidth=2)
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax5.set_title(f'Difference (MSE: {mse:.6f})', fontsize=12, fontweight='bold')
ax5.set_xlabel('Wavenumber Index', fontsize=10)
ax5.set_ylabel('Difference', fontsize=10)
ax5.grid(True, alpha=0.3)

plt.suptitle('WGAN Training Results Summary', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(os.path.join(output_dir, 'SUMMARY.png'), dpi=300, bbox_inches='tight')
print(f"   Saved: SUMMARY.png")
plt.close()

print(f"\n{'='*60}")
print(f"REPORT GENERATION COMPLETE!")
print(f"{'='*60}")
print(f"\nAll results saved to: {output_dir}/")
print(f"\nContents:")
print(f"  - visualizations/")
print(f"      * comparison_real_vs_generated.png")
print(f"      * statistical_analysis.png")
print(f"      * training_progression.gif")
print(f"  - models/")
print(f"      * generator.pkl")
print(f"      * discriminator.pkl")
print(f"  - training_images/ (64 epoch images)")
print(f"  - generated_samples/ (100 synthetic signals)")
print(f"  - SUMMARY.png")
print(f"\n{'='*60}\n")
