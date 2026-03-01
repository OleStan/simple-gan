
import torch
import torch.optim as optim
import numpy as np
import json
import os
import argparse
import time
from datetime import datetime
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler

from model import (
    ContinuousConditionalGenerator, SpectralNormConv1DCriticV3,
    weights_init, GeneratorEMA, JacobianRegularizer
)
# Re-use from v2
from models.improved_wgan_v2.model import PhysicsInformedLossV2, compute_gradient_penalty

class ProfileDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.X = np.load(data_path)
        with open(Path(data_path).parent / 'normalization_params.json', 'r') as f:
            self.norm_params = json.load(f)
        self.K = self.norm_params['K']
        labels_path = Path(data_path).parent / 'y_labels.npy'
        self.labels = np.load(labels_path).astype(np.int64)
        
        # Normalize to [-1, 1]
        self.X_norm = self.X.copy()
        s_min, s_max = self.norm_params['sigma_min'], self.norm_params['sigma_max']
        m_min, m_max = self.norm_params['mu_min'], self.norm_params['mu_max']
        self.X_norm[:, :self.K] = 2 * (self.X[:, :self.K] - s_min) / (s_max - s_min + 1e-8) - 1
        self.X_norm[:, self.K:] = 2 * (self.X[:, self.K:] - m_min) / (m_max - m_min + 1e-8) - 1

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X_norm[idx]), self.labels[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nz', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load dataset to get class count
    dataset = ProfileDataset('../../data/training/X_raw.npy')
    n_classes = dataset.norm_params['n_classes']
    K = dataset.K
    
    # Init Models
    netG = ContinuousConditionalGenerator(nz=args.nz, K=K, n_classes=n_classes).to(device)
    netC = SpectralNormConv1DCriticV3(K=K, n_classes=n_classes).to(device)
    netG.apply(weights_init)
    netC.apply(weights_init)
    
    ema = GeneratorEMA(netG)
    jacobian_reg = JacobianRegularizer(lambda_jacobian=0.05)
    physics_loss_fn = PhysicsInformedLossV2()
    
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerC = optim.Adam(netC.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Cosine Annealing for high-precision convergence
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=args.epochs)
    schedulerC = optim.lr_scheduler.CosineAnnealingLR(optimizerC, T_max=args.epochs)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    scaler = GradScaler()

    # Output Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = os.environ.get('RESULT_TAG', 'v3_test')
    output_dir = Path(f"../../results/{tag}/improved_wgan_v3_nz{args.nz}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    print(f"Starting V3 Training: {args.nz} nz, {args.epochs} epochs")

    for epoch in range(args.epochs):
        for i, (real_data, labels) in enumerate(dataloader):
            real_data, labels = real_data.to(device), labels.to(device)
            b_size = real_data.size(0)
            
            # 1. Train Critic
            optimizerC.zero_grad()
            with autocast():
                output_real = netC(real_data, labels=labels)
                noise = torch.randn(b_size, args.nz, device=device)
                gen_labels = torch.randint(0, n_classes, (b_size,), device=device)
                fake_data, _, _ = netG(noise, labels=gen_labels)
                output_fake = netC(fake_data.detach(), labels=gen_labels)
                c_loss = output_fake.mean() - output_real.mean()
            
            scaler.scale(c_loss).backward()
            scaler.step(optimizerC)
            
            # 2. Train Generator
            optimizerG.zero_grad()
            with autocast():
                output_fake = netC(fake_data, labels=gen_labels)
                g_adv = -output_fake.mean()
                
                # V3 Physics & Smoothness
                _, sigma_f, mu_f = netG(noise, labels=gen_labels)
                p_loss, _ = physics_loss_fn(sigma_f, mu_f, epoch=epoch, max_epochs=args.epochs)
                j_loss = jacobian_reg(netG, noise, labels=gen_labels)
                
                total_g = g_adv + p_loss + j_loss
                
            scaler.scale(total_g).backward()
            scaler.step(optimizerG)
            scaler.update()
            
            ema.update()

        schedulerG.step()
        schedulerC.step()
        
        if epoch % 500 == 0 or epoch == args.epochs - 1:
            print(f"[{epoch}/{args.epochs}] Loss_C: {c_loss.item():.4f} Loss_G: {total_g.item():.4f}")
            # Save EMA version as the primary
            ema.apply_shadow()
            torch.save(netG.state_dict(), output_dir / "models" / f"netG_epoch_{epoch}.pt")
            ema.restore()

    # Final Save
    ema.apply_shadow()
    torch.save(netG.state_dict(), output_dir / "models" / "netG_final.pt")
    print(f"Training Complete. Results in {output_dir}")

if __name__ == "__main__":
    main()
