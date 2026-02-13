import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Conv1DGenerator(nn.Module):
    """
    Improved generator using 1D convolutions for spatial coherence.
    Generates σ and μ profiles as continuous signals.
    """
    def __init__(self, nz=100, K=50, ngf=256):
        super().__init__()
        self.K = K
        self.ngf = ngf
        
        self.fc = nn.Linear(nz, ngf * 4)
        
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
        )
        
        self.sigma_head = nn.Sequential(
            nn.Conv1d(ngf // 4, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(32, 1, 3, 1, 1),
            nn.Tanh()
        )
        
        self.mu_head = nn.Sequential(
            nn.Conv1d(ngf // 4, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(32, 1, 3, 1, 1),
            nn.Tanh()
        )
        
        self.final_adjust = nn.AdaptiveAvgPool1d(K)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.ngf, 4)
        
        features = self.upsample(x)
        
        sigma_raw = self.sigma_head(features)
        mu_raw = self.mu_head(features)
        
        sigma_out = self.final_adjust(sigma_raw).squeeze(1)
        mu_out = self.final_adjust(mu_raw).squeeze(1)
        
        output = torch.cat([sigma_out, mu_out], dim=1)
        
        return output, sigma_out, mu_out


class Conv1DCritic(nn.Module):
    """
    Improved critic using 1D convolutions to evaluate spatial patterns.
    """
    def __init__(self, K=50, ndf=128):
        super().__init__()
        self.K = K
        
        self.sigma_encoder = nn.Sequential(
            nn.Conv1d(1, ndf // 2, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf // 2, ndf, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf, ndf, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.mu_encoder = nn.Sequential(
            nn.Conv1d(1, ndf // 2, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf // 2, ndf, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf, ndf, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.joint_processor = nn.Sequential(
            nn.Conv1d(ndf * 2, ndf * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        sigma = x[:, :self.K].unsqueeze(1)
        mu = x[:, self.K:].unsqueeze(1)
        
        sigma_features = self.sigma_encoder(sigma)
        mu_features = self.mu_encoder(mu)
        
        combined = torch.cat([sigma_features, mu_features], dim=1)
        
        joint_features = self.joint_processor(combined)
        
        output = self.fc(joint_features)
        
        return output


class PhysicsInformedLoss(nn.Module):
    """
    Additional loss components for physical plausibility.
    """
    def __init__(self, lambda_smooth=0.1, lambda_bounds=0.05):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_bounds = lambda_bounds
    
    def smoothness_loss(self, profile):
        """
        Penalize rapid changes (non-smooth profiles).
        Computes L2 norm of first differences.
        """
        diff = profile[:, 1:] - profile[:, :-1]
        return torch.mean(diff ** 2)
    
    def bounds_penalty(self, profile, min_val=-1.0, max_val=1.0):
        """
        Soft penalty for values outside expected range.
        """
        lower_violation = F.relu(min_val - profile)
        upper_violation = F.relu(profile - max_val)
        return torch.mean(lower_violation ** 2 + upper_violation ** 2)
    
    def forward(self, sigma_profile, mu_profile):
        """
        Compute combined physics-informed loss.
        """
        smooth_sigma = self.smoothness_loss(sigma_profile)
        smooth_mu = self.smoothness_loss(mu_profile)
        
        bounds_sigma = self.bounds_penalty(sigma_profile)
        bounds_mu = self.bounds_penalty(mu_profile)
        
        total_loss = (
            self.lambda_smooth * (smooth_sigma + smooth_mu) +
            self.lambda_bounds * (bounds_sigma + bounds_mu)
        )
        
        return total_loss, {
            'smooth_sigma': smooth_sigma.item(),
            'smooth_mu': smooth_mu.item(),
            'bounds_sigma': bounds_sigma.item(),
            'bounds_mu': bounds_mu.item()
        }


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """
    Compute gradient penalty for WGAN-GP.
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    
    fake = torch.ones(batch_size, 1, device=device, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


class ProfileQualityMetrics:
    """
    Metrics to evaluate generated profile quality.
    """
    @staticmethod
    def compute_smoothness_score(profile):
        """
        Higher score = smoother profile.
        Based on inverse of gradient variance.
        """
        diff = profile[:, 1:] - profile[:, :-1]
        variance = torch.var(diff, dim=1)
        score = 1.0 / (1.0 + variance)
        return score.mean().item()
    
    @staticmethod
    def compute_monotonicity_score(profile):
        """
        Fraction of profiles that are monotonic.
        """
        diff = profile[:, 1:] - profile[:, :-1]
        monotone_increasing = (diff >= 0).all(dim=1).float()
        monotone_decreasing = (diff <= 0).all(dim=1).float()
        monotone = monotone_increasing + monotone_decreasing
        return monotone.mean().item()
    
    @staticmethod
    def compute_diversity_score(profiles):
        """
        Measure diversity using pairwise distances.
        """
        n = profiles.size(0)
        if n < 2:
            return 0.0
        
        dists = torch.cdist(profiles, profiles, p=2)
        mask = ~torch.eye(n, dtype=torch.bool, device=profiles.device)
        avg_dist = dists[mask].mean()
        
        return avg_dist.item()
    
    @staticmethod
    def evaluate_batch(sigma, mu):
        """
        Comprehensive evaluation of a batch.
        """
        return {
            'sigma_smoothness': ProfileQualityMetrics.compute_smoothness_score(sigma),
            'mu_smoothness': ProfileQualityMetrics.compute_smoothness_score(mu),
            'sigma_monotonicity': ProfileQualityMetrics.compute_monotonicity_score(sigma),
            'mu_monotonicity': ProfileQualityMetrics.compute_monotonicity_score(mu),
            'sigma_diversity': ProfileQualityMetrics.compute_diversity_score(sigma),
            'mu_diversity': ProfileQualityMetrics.compute_diversity_score(mu)
        }
