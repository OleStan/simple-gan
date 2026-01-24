import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os

from preprocessing_mddect import Dataset, DatasetNpy
from wgan_mddect import Generator, Discriminator, weights_init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description='Train WGAN on MDDECT signals (txt or npy)')
parser.add_argument('--dataset-type', type=str, default='txt', choices=['txt', 'npy'], help='Dataset type: txt (extracted files) or npy (direct .npy file)')
parser.add_argument('--data-root', type=str, default='./data/mddect', help='Folder with extracted .txt signals (for txt mode) or path to .npy file (for npy mode)')
parser.add_argument('--img-dir', type=str, default='./img_mddect', help='Folder to save epoch images')
parser.add_argument('--nets-dir', type=str, default='./nets_mddect', help='Folder to save trained models')
parser.add_argument('--signal-length', type=int, default=1024, help='Target signal length after preprocessing')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=64)
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--clip-value', type=float, default=0.01)
parser.add_argument('--n-critic', type=int, default=5)
parser.add_argument('--classes', type=int, nargs='+', default=None, help='Class indices to use (npy mode only, 0-19). Default: all')
parser.add_argument('--angles', type=int, nargs='+', default=None, help='Scanning angle indices to use (npy mode only, 0-7). Default: all')
parser.add_argument('--channel', type=int, default=0, choices=[0, 1], help='Channel to use (npy mode only, 0 or 1). Default: 0')
args = parser.parse_args()

os.makedirs(args.img_dir, exist_ok=True)
os.makedirs(args.nets_dir, exist_ok=True)

print("Loading MDDECT dataset...")
if args.dataset_type == 'npy':
    dataset = DatasetNpy(
        npy_file=args.data_root,
        target_length=args.signal_length,
        classes=args.classes,
        angles=args.angles,
        channel=args.channel
    )
    print(f"Dataset loaded from .npy: {len(dataset)} samples")
    if args.classes:
        print(f"  Classes: {args.classes}")
    if args.angles:
        print(f"  Angles: {args.angles}")
    print(f"  Channel: {args.channel}")
else:
    dataset = Dataset(args.data_root, target_length=args.signal_length)
    print(f"Dataset loaded from txt files: {len(dataset)} samples")

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

netG = Generator(args.nz).to(device)
netG.apply(weights_init)

netD = Discriminator().to(device)
netD.apply(weights_init)

optimizerD = optim.RMSprop(netD.parameters(), lr=args.lr)
optimizerG = optim.RMSprop(netG.parameters(), lr=args.lr)

fixed_noise = torch.randn(args.batch_size, args.nz, 1, device=device)

print("\nStarting WGAN training on MDDECT dataset...")
print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Signal length: {args.signal_length}")
print(f"Learning rate: {args.lr}, Clip value: {args.clip_value}, n_critic: {args.n_critic}\n")

for epoch in range(args.epochs):
    for i, (data, _) in enumerate(dataloader):
        
        netD.zero_grad()
        real = data.to(device)
        batch_size_current = real.size(0)
        
        output_real = netD(real).view(-1)
        D_real = output_real.mean()
        
        noise = torch.randn(batch_size_current, args.nz, 1, device=device)
        fake = netG(noise)
        output_fake = netD(fake.detach()).view(-1)
        D_fake = output_fake.mean()
        
        D_loss = -(D_real - D_fake)
        D_loss.backward()
        optimizerD.step()
        
        for p in netD.parameters():
            p.data.clamp_(-args.clip_value, args.clip_value)
        
        if i % args.n_critic == 0:
            netG.zero_grad()
            output = netD(fake).view(-1)
            G_loss = -output.mean()
            G_loss.backward()
            optimizerG.step()
        
        if i % 5 == 0:
            print(f'[{epoch}/{args.epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {D_loss.item():.4f} Loss_G: {G_loss.item():.4f}')
    
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    
    plt.figure(figsize=(12, 6))
    for j in range(min(8, args.batch_size)):
        plt.plot(fake[j, 0, :].numpy(), alpha=0.7, linewidth=1)
    plt.title(f'Generated MDDECT Signals - Epoch {epoch}')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.img_dir, f'wgan_mddect_epoch_{epoch}.png'))
    plt.close()

torch.save(netG, os.path.join(args.nets_dir, 'wgan_mddect_netG.pkl'))
torch.save(netD, os.path.join(args.nets_dir, 'wgan_mddect_netD.pkl'))

print("\nTraining complete!")
print(f"Models saved to {args.nets_dir}/")
print(f"Training images saved to {args.img_dir}/")
