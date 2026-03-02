
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import copy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

_EMBED_DIM = 32

class ContinuousConditionalGenerator(nn.Module):
    """
    Improved Generator v3: Inverse-Solver-Ready (ISR).
    - Continuous class steering via a dense layer (no more hard lookup).
    - Architecture tuned for latent space smoothness.
    """
    def __init__(self, nz=100, K=50, ngf=256, n_classes=1):
        super().__init__()
        self.K = K
        self.ngf = ngf
        self.n_classes = n_classes
        
        # Class embedding: Use a small MLP instead of nn.Embedding 
        # to allow for "soft" continuous labels during inversion.
        if n_classes > 1:
            self.class_mlp = nn.Sequential(
                nn.Linear(n_classes, _EMBED_DIM),
                nn.LeakyReLU(0.2, inplace=True)
            )
            input_dim = nz + _EMBED_DIM
        else:
            input_dim = nz

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
        
        # Refined heads for 1D signal clarity
        self.sigma_head = nn.Sequential(
            nn.Conv1d(ngf // 4, 64, 5, 1, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 1, 3, 1, 1),
            nn.Tanh()
        )
        
        self.mu_head = nn.Sequential(
            nn.Conv1d(ngf // 4, 64, 5, 1, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 1, 3, 1, 1),
            nn.Tanh()
        )
        
        self.final_adjust = nn.AdaptiveAvgPool1d(K)
    
    def forward(self, z, labels=None):
        """
        labels: If n_classes > 1, expects one-hot encoded vector (batch, n_classes)
                to allow for continuous steering between classes.
        """
        if self.n_classes > 1 and labels is not None:
            # If labels are indices, convert to one-hot automatically
            if labels.dim() == 1 or (labels.dim() == 2 and labels.size(1) == 1):
                labels_oh = F.one_hot(labels.long().view(-1), self.n_classes).float()
            else:
                labels_oh = labels
            
            emb = self.class_mlp(labels_oh)
            z = torch.cat([z, emb], dim=1)
        
        x = self.fc(z)
        x = x.view(-1, self.ngf, 4)
        
        features = self.upsample(x)
        
        sigma_raw = self.sigma_head(features)
        mu_raw = self.mu_head(features)
        
        sigma_out = self.final_adjust(sigma_raw).squeeze(1)
        mu_out = self.final_adjust(mu_raw).squeeze(1)
        
        output = torch.cat([sigma_out, mu_out], dim=1)
        return output, sigma_out, mu_out

class JacobianRegularizer(nn.Module):
    """
    Enforces latent space smoothness by penalizing the norm of the Jacobian.
    Crucial for high-stability inverse solvers.
    """
    def __init__(self, lambda_jacobian=0.1):
        super().__init__()
        self.lambda_jacobian = lambda_jacobian

    def forward(self, G, z, labels=None):
        z.requires_grad_(True)
        # We only need the mean behavior for speed
        output, _, _ = G(z, labels=labels)
        
        # Random projection for fast Jacobian norm estimation
        v = torch.randn_like(output)
        v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)
        
        # Calculate v^T * J
        v_J = torch.autograd.grad(output, z, grad_outputs=v, create_graph=True, retain_graph=True)[0]
        
        # Loss is the norm of the gradient
        loss = v_J.pow(2).sum(dim=1).mean()
        return self.lambda_jacobian * loss

class OrthogonalRegularizer(nn.Module):
    """
    Forces generator weights to be orthogonal.
    Prevents mode collapse and ensures latent dimensions are independent.
    """
    def __init__(self, lambda_ortho=1e-4):
        super().__init__()
        self.lambda_ortho = lambda_ortho

    def forward(self, model):
        loss = 0
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                # For Linear and Conv layers
                w = param.view(param.shape[0], -1)
                sym = torch.mm(w, w.t())
                identity = torch.eye(w.shape[0], device=w.device)
                loss += torch.norm(sym - identity)
        return self.lambda_ortho * loss

class GeneratorEMA:
    """
    Exponential Moving Average of Generator weights.
    Provides a low-noise version of the model for production/inversion.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

# Reuse Critic and Physics Loss from v2 with minor tweaks
class SpectralNormConv1DCriticV3(nn.Module):
    def __init__(self, K=50, ndf=128, n_classes=1):
        super().__init__()
        self.K = K
        self.n_classes = n_classes

        # Critic also uses dense embedding for labels to match generator
        if n_classes > 1:
            self.class_mlp = nn.Sequential(
                nn.Linear(n_classes, _EMBED_DIM),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.sigma_encoder = nn.Sequential(
            spectral_norm(nn.Conv1d(1, ndf // 2, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(ndf // 2, ndf, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(ndf, ndf, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mu_encoder = nn.Sequential(
            spectral_norm(nn.Conv1d(1, ndf // 2, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(ndf // 2, ndf, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(ndf, ndf, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.joint_processor = nn.Sequential(
            spectral_norm(nn.Conv1d(ndf * 2, ndf * 2, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        fc_input_dim = ndf * 2 + _EMBED_DIM if n_classes > 1 else ndf * 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            spectral_norm(nn.Linear(fc_input_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            spectral_norm(nn.Linear(256, 1))
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
            if labels.dim() == 1 or labels.size(1) == 1:
                labels_oh = F.one_hot(labels.long().view(-1), self.n_classes).float()
            else:
                labels_oh = labels
            emb = self.class_mlp(labels_oh)
            flat = torch.cat([flat, emb], dim=1)

        return self.fc(flat)
