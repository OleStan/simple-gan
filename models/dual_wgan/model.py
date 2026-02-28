import torch
import torch.nn as nn

_EMBED_DIM = 32


class Critic(nn.Module):
    def __init__(self, input_dim=100, n_classes=1):
        super(Critic, self).__init__()
        self.n_classes = n_classes

        effective_dim = input_dim + _EMBED_DIM if n_classes > 1 else input_dim
        if n_classes > 1:
            self.label_embedding = nn.Embedding(n_classes, _EMBED_DIM)

        self.main = nn.Sequential(
            nn.Linear(effective_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 1)
        )

    def forward(self, x, labels=None):
        if self.n_classes > 1 and labels is not None:
            emb = self.label_embedding(labels)
            x = torch.cat([x, emb], dim=1)
        return self.main(x)


class DualHeadGenerator(nn.Module):
    def __init__(self, nz=100, K=50, n_classes=1):
        super(DualHeadGenerator, self).__init__()
        self.nz = nz
        self.K = K
        self.n_classes = n_classes

        encoder_input_dim = nz + _EMBED_DIM if n_classes > 1 else nz
        if n_classes > 1:
            self.label_embedding = nn.Embedding(n_classes, _EMBED_DIM)

        self.shared_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )

        self.sigma_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, K),
            nn.Tanh()
        )

        self.mu_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, K),
            nn.Tanh()
        )

    def forward(self, z, labels=None):
        if self.n_classes > 1 and labels is not None:
            emb = self.label_embedding(labels)
            z = torch.cat([z, emb], dim=1)

        shared_features = self.shared_encoder(z)
        sigma_out = self.sigma_head(shared_features)
        mu_out = self.mu_head(shared_features)
        combined = torch.cat([sigma_out, mu_out], dim=1)

        return combined, sigma_out, mu_out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def compute_gradient_penalty(critic, real_data, fake_data, device, labels=None):
    batch_size = real_data.size(0)

    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True)

    disc_interpolates = critic(interpolates, labels)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    K = 50
    nz = 100
    n_classes = 2
    batch_size = 32

    netG = DualHeadGenerator(nz=nz, K=K, n_classes=n_classes).to(device)
    netC = Critic(input_dim=2*K, n_classes=n_classes).to(device)

    netG.apply(weights_init)
    netC.apply(weights_init)

    print("Generator:")
    print(netG)
    print(f"\nTotal parameters: {sum(p.numel() for p in netG.parameters()):,}")

    print("\n" + "="*60)
    print("Critic:")
    print(netC)
    print(f"\nTotal parameters: {sum(p.numel() for p in netC.parameters()):,}")

    noise = torch.randn(batch_size, nz, device=device)
    labels = torch.randint(0, n_classes, (batch_size,), device=device)
    combined, sigma, mu = netG(noise, labels)

    print("\n" + "="*60)
    print("Test forward pass:")
    print(f"Input noise shape: {noise.shape}")
    print(f"Combined output shape: {combined.shape}")
    print(f"Sigma output shape: {sigma.shape}")
    print(f"Mu output shape: {mu.shape}")

    critic_out = netC(combined, labels)
    print(f"Critic output shape: {critic_out.shape}")
