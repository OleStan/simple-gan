import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


_EMBED_DIM = 32


class ConditionalConv1DGenerator(nn.Module):
    """
    Improved generator v2 with:
    - Spectral normalization for stability
    - Optional conditional generation
    - Better architecture for smooth profiles
    """
    def __init__(self, nz=100, K=50, ngf=256, conditional=False, n_conditions=0, n_classes=1):
        super().__init__()
        self.K = K
        self.ngf = ngf
        self.conditional = conditional
        self.n_classes = n_classes

        embed_dim = _EMBED_DIM if n_classes > 1 else n_conditions
        input_dim = nz + embed_dim if (conditional or n_classes > 1) else nz
        if n_classes > 1:
            self.label_embedding = nn.Embedding(n_classes, _EMBED_DIM)

        self.fc = nn.Linear(input_dim, ngf * 4)
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(ngf, ngf, 4, 2, 1),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(ngf, ngf // 2, 4, 2, 1),
            nn.BatchNorm1d(ngf // 2),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(ngf // 2, ngf // 4, 4, 2, 1),
            nn.BatchNorm1d(ngf // 4),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(ngf // 4, ngf // 4, 4, 2, 1),
            nn.BatchNorm1d(ngf // 4),
            nn.ReLU(True),
        )
        
        self.sigma_head = nn.Sequential(
            nn.Conv1d(ngf // 4, 64, 5, 1, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(32, 1, 3, 1, 1),
            nn.Tanh()
        )
        
        self.mu_head = nn.Sequential(
            nn.Conv1d(ngf // 4, 64, 5, 1, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(32, 1, 3, 1, 1),
            nn.Tanh()
        )
        
        self.final_adjust = nn.AdaptiveAvgPool1d(K)
    
    def forward(self, z, conditions=None, labels=None):
        if self.n_classes > 1 and labels is not None:
            emb = self.label_embedding(labels)
            z = torch.cat([z, emb], dim=1)
        elif self.conditional and conditions is not None:
            z = torch.cat([z, conditions], dim=1)
        
        x = self.fc(z)
        x = x.view(-1, self.ngf, 4)
        
        features = self.upsample(x)
        
        sigma_raw = self.sigma_head(features)
        mu_raw = self.mu_head(features)
        
        sigma_out = self.final_adjust(sigma_raw).squeeze(1)
        mu_out = self.final_adjust(mu_raw).squeeze(1)
        
        output = torch.cat([sigma_out, mu_out], dim=1)
        
        return output, sigma_out, mu_out


class SpectralNormConv1DCritic(nn.Module):
    """
    Improved critic v2 with spectral normalization for stability.
    No gradient penalty needed when using spectral norm.
    """
    def __init__(self, K=50, ndf=128, use_spectral_norm=True, n_classes=1):
        super().__init__()
        self.K = K
        self.use_spectral_norm = use_spectral_norm
        self.n_classes = n_classes

        def maybe_spectral_norm(layer):
            return spectral_norm(layer) if use_spectral_norm else layer

        if n_classes > 1:
            self.label_embedding = nn.Embedding(n_classes, _EMBED_DIM)

        self.sigma_encoder = nn.Sequential(
            maybe_spectral_norm(nn.Conv1d(1, ndf // 2, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),

            maybe_spectral_norm(nn.Conv1d(ndf // 2, ndf, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),

            maybe_spectral_norm(nn.Conv1d(ndf, ndf, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mu_encoder = nn.Sequential(
            maybe_spectral_norm(nn.Conv1d(1, ndf // 2, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),

            maybe_spectral_norm(nn.Conv1d(ndf // 2, ndf, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),

            maybe_spectral_norm(nn.Conv1d(ndf, ndf, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.joint_processor = nn.Sequential(
            maybe_spectral_norm(nn.Conv1d(ndf * 2, ndf * 2, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        fc_input_dim = ndf * 2 + _EMBED_DIM if n_classes > 1 else ndf * 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            maybe_spectral_norm(nn.Linear(fc_input_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            maybe_spectral_norm(nn.Linear(256, 1))
        )

    def forward(self, x, labels=None):
        sigma = x[:, :self.K].unsqueeze(1)
        mu = x[:, self.K:].unsqueeze(1)

        sigma_features = self.sigma_encoder(sigma)
        mu_features = self.mu_encoder(mu)

        combined = torch.cat([sigma_features, mu_features], dim=1)
        joint_features = self.joint_processor(combined)
        flat = joint_features.view(joint_features.size(0), -1)

        if self.n_classes > 1 and labels is not None:
            emb = self.label_embedding(labels)
            flat = torch.cat([flat, emb], dim=1)

        return self.fc(flat)


class PhysicsInformedLossV2(nn.Module):
    """
    Improved physics-informed loss with better stability.
    Uses softer penalties and optional scheduling.
    """
    def __init__(self, lambda_smooth=0.05, lambda_bounds=0.02, lambda_monotonic=0.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_bounds = lambda_bounds
        self.lambda_monotonic = lambda_monotonic
    
    def smoothness_loss(self, profile):
        diff = profile[:, 1:] - profile[:, :-1]
        return torch.mean(diff ** 2)
    
    def bounds_penalty(self, profile):
        penalty = torch.mean(F.relu(profile - 1.0) + F.relu(-1.0 - profile))
        return penalty
    
    def monotonic_penalty(self, profile, increasing=True):
        diff = profile[:, 1:] - profile[:, :-1]
        if increasing:
            violation = F.relu(-diff)
        else:
            violation = F.relu(diff)
        return torch.mean(violation)
    
    def forward(self, sigma, mu, epoch=None, max_epochs=None):
        schedule_factor = 1.0
        if epoch is not None and max_epochs is not None:
            warmup_epochs = min(50, max_epochs // 10)
            if epoch < warmup_epochs:
                schedule_factor = epoch / warmup_epochs
        
        smooth_sigma = self.smoothness_loss(sigma)
        smooth_mu = self.smoothness_loss(mu)
        
        bounds_sigma = self.bounds_penalty(sigma)
        bounds_mu = self.bounds_penalty(mu)
        
        total_loss = (
            self.lambda_smooth * (smooth_sigma + smooth_mu) * schedule_factor +
            self.lambda_bounds * (bounds_sigma + bounds_mu) * schedule_factor
        )
        
        if self.lambda_monotonic > 0:
            mono_sigma = self.monotonic_penalty(sigma, increasing=True)
            total_loss += self.lambda_monotonic * mono_sigma * schedule_factor
        
        metrics = {
            'smooth_sigma': smooth_sigma.item(),
            'smooth_mu': smooth_mu.item(),
            'bounds_sigma': bounds_sigma.item(),
            'bounds_mu': bounds_mu.item(),
            'schedule_factor': schedule_factor
        }
        
        return total_loss, metrics


class ProfileQualityMetrics:
    """
    Quality metrics for evaluating generated profiles.
    """
    @staticmethod
    def smoothness(profile):
        diff = np.diff(profile, axis=1)
        return 1.0 / (1.0 + np.mean(diff ** 2, axis=1).mean())
    
    @staticmethod
    def monotonicity(profile):
        diff = np.diff(profile, axis=1)
        increasing = (diff >= 0).sum() / diff.size
        return max(increasing, 1 - increasing)
    
    @staticmethod
    def diversity(profiles):
        pairwise_dist = []
        n = len(profiles)
        for i in range(min(n, 100)):
            for j in range(i+1, min(n, 100)):
                dist = np.linalg.norm(profiles[i] - profiles[j])
                pairwise_dist.append(dist)
        return np.mean(pairwise_dist) if pairwise_dist else 0.0


def compute_gradient_penalty(critic, real_data, fake_data, device, lambda_gp=10):
    """
    Gradient penalty for WGAN-GP.
    Only needed if not using spectral normalization.
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_data)
    
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True)
    
    disc_interpolates = critic(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    
    return gradient_penalty


import numpy as np
