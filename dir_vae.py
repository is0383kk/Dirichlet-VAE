"""
Dirichlet Variational Auto-Encoder (Dir-VAE) implementation for MNIST
Based on "Autoencodeing Variational Inference for Topic Model" (ICLR2017)
"""

from __future__ import print_function

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


@dataclass
class Config:
    """Configuration class for Dir-VAE training"""

    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-3
    no_cuda: bool = False
    seed: int = 10
    log_interval: int = 2
    category: int = 10  # Number of latent categories (K)
    alpha: float = 0.3  # Dirichlet hyperparameter
    data_dir: str = "./data"
    output_dir: str = "./image"

    # Network architecture parameters
    encoder_channels: int = 64
    decoder_channels: int = 64
    input_channels: int = 1
    latent_dim: int = 1024
    hidden_dim: int = 512


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(description="Dir-VAE MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disable CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--category",
        type=int,
        default=10,
        metavar="K",
        help="the number of categories in the dataset",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Dirichlet hyperparameter alpha (default: 0.3)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="directory for dataset (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./image",
        help="directory for output images (default: ./image)",
    )
    return parser


def setup_device_and_seed(config: Config) -> torch.device:
    """Setup device and random seeds"""
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Determine device
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Using device: {device}")
    return device


def create_data_loaders(
    config: Config,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test data loaders"""
    # Create data directory if it doesn't exist
    os.makedirs(config.data_dir, exist_ok=True)

    # Data loader kwargs
    kwargs = (
        {"num_workers": 1, "pin_memory": True}
        if not config.no_cuda and torch.cuda.is_available()
        else {}
    )

    # Create datasets and loaders
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            config.data_dir, train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=config.batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(config.data_dir, train=False, transform=transforms.ToTensor()),
        batch_size=config.batch_size,
        shuffle=False,
        **kwargs,
    )

    return train_loader, test_loader


def compute_dirichlet_prior(
    K: int, alpha: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Dirichlet prior parameters using Laplace approximation.

    Args:
        K: Number of categories
        alpha: Dirichlet hyperparameter
        device: torch device

    Returns:
        Tuple of (mean, variance) tensors for the approximated normal distribution
    """
    # Laplace approximation to convert Dirichlet to multivariate normal
    a = torch.full((1, K), alpha, dtype=torch.float, device=device)
    mean = a.log().t() - a.log().mean(1, keepdim=True)
    var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K**2) * a.reciprocal().sum(
        1, keepdim=True
    )
    return mean.t(), var.t()


class DirVAEEncoder(nn.Module):
    """Encoder part of Dir-VAE"""

    def __init__(self, config: Config):
        super(DirVAEEncoder, self).__init__()
        ndf = config.encoder_channels
        nc = config.input_channels

        self.conv_layers = nn.Sequential(
            # input: (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, config.latent_dim, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc1 = nn.Linear(config.latent_dim, config.hidden_dim)
        self.fc_mu = nn.Linear(config.hidden_dim, config.category)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.category)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv_layers(x)
        h1 = self.fc1(conv_out.view(conv_out.size(0), -1))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar


class DirVAEDecoder(nn.Module):
    """Decoder part of Dir-VAE"""

    def __init__(self, config: Config):
        super(DirVAEDecoder, self).__init__()
        ngf = config.decoder_channels
        nc = config.input_channels

        self.fc_decode = nn.Linear(config.category, config.hidden_dim)
        self.fc_deconv = nn.Linear(config.hidden_dim, config.latent_dim)

        self.deconv_layers = nn.Sequential(
            # input: latent_dim x 1 x 1
            nn.ConvTranspose2d(config.latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            # output: (nc) x 32 x 32
        )

        self.relu = nn.ReLU()

    def forward(self, gauss_z: torch.Tensor) -> torch.Tensor:
        # Apply softmax to satisfy simplex constraint (Dirichlet distribution)
        dir_z = F.softmax(gauss_z, dim=1)

        h3 = self.relu(self.fc_decode(dir_z))
        deconv_input = self.fc_deconv(h3)
        deconv_input = deconv_input.view(-1, deconv_input.size(1), 1, 1)

        return self.deconv_layers(deconv_input)


class DirVAE(nn.Module):
    """Dirichlet Variational Auto-Encoder"""

    def __init__(self, config: Config, device: torch.device):
        super(DirVAE, self).__init__()
        self.config = config
        self.device = device

        # Initialize encoder and decoder
        self.encoder = DirVAEEncoder(config)
        self.decoder = DirVAEDecoder(config)

        # Setup Dirichlet prior
        self._setup_dirichlet_prior()

    def _setup_dirichlet_prior(self):
        """Setup Dirichlet prior parameters"""
        prior_mean, prior_var = compute_dirichlet_prior(
            self.config.category, self.config.alpha, self.device
        )
        self.register_buffer("prior_mean", prior_mean)
        self.register_buffer("prior_var", prior_var)
        self.register_buffer("prior_logvar", prior_var.log())

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters"""
        return self.encoder(x)

    def decode(self, gauss_z: torch.Tensor) -> torch.Tensor:
        """Decode latent variables to reconstruction"""
        return self.decoder(gauss_z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backpropagation through stochastic nodes"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        mu, logvar = self.encode(x)
        gauss_z = self.reparameterize(mu, logvar)

        # gauss_z follows multivariate normal distribution
        # Applying softmax gives us Dirichlet-distributed variables
        dir_z = F.softmax(gauss_z, dim=1)
        recon_x = self.decode(gauss_z)

        return recon_x, mu, logvar, gauss_z, dir_z

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss function: reconstruction loss + KL divergence

        Args:
            recon_x: Reconstructed images
            x: Original images
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Total loss (sum over batch)
        """
        # Reconstruction loss (Binary Cross Entropy)
        BCE = F.binary_cross_entropy(
            recon_x.view(-1, 784), x.view(-1, 784), reduction="sum"
        )

        # KL divergence between Dirichlet prior and variational posterior
        # Based on the original paper: "Autoencodeing variational inference for topic model"
        prior_mean = self.prior_mean.expand_as(mu)
        prior_var = self.prior_var.expand_as(logvar)
        prior_logvar = self.prior_logvar.expand_as(logvar)

        var_division = logvar.exp() / prior_var  # Σ_0 / Σ_1
        diff = mu - prior_mean  # μ_1 - μ_0
        diff_term = diff * diff / prior_var  # (μ_1 - μ_0)² / Σ_1
        logvar_division = prior_logvar - logvar  # log|Σ_1| - log|Σ_0|

        # KL divergence
        KLD = 0.5 * (
            var_division + diff_term + logvar_division - self.config.category
        ).sum(dim=1)

        return BCE + KLD.sum()


class DirVAETrainer:
    """Trainer class for Dir-VAE"""

    def __init__(
        self,
        model: DirVAE,
        config: Config,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)

            self.optimizer.zero_grad()
            recon_batch, mu, logvar, gauss_z, dir_z = self.model(data)
            loss = self.model.loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            if batch_idx % self.config.log_interval == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data):5d}/"
                    f"{len(self.train_loader.dataset)} "
                    f"({100. * batch_idx / len(self.train_loader):3.0f}%)]"
                    f"\tLoss: {loss.item() / len(data):.6f}"
                )

        avg_loss = train_loss / len(self.train_loader.dataset)
        print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
        return avg_loss

    def test_epoch(self, epoch: int) -> float:
        """Test for one epoch"""
        self.model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar, gauss_z, dir_z = self.model(data)
                loss = self.model.loss_function(recon_batch, data, mu, logvar)
                test_loss += loss.item()

                if i == 0:
                    n = min(data.size(0), 18)
                    # Properly reshape for comparison
                    comparison = torch.cat(
                        [data[:n], recon_batch.view(data.size(0), 1, 28, 28)[:n]]
                    )
                    save_image(
                        comparison.cpu(),
                        os.path.join(self.config.output_dir, f"recon_{epoch}.png"),
                        nrow=n,
                    )

        avg_loss = test_loss / len(self.test_loader.dataset)
        print(f"====> Test set loss: {avg_loss:.4f}")
        return avg_loss

    def generate_samples(self, epoch: int, num_samples: int = 64):
        """Generate samples from the model"""
        self.model.eval()
        with torch.no_grad():
            # Sample from latent space
            sample = torch.randn(num_samples, self.config.category).to(self.device)
            sample = self.model.decode(sample).cpu()
            save_image(
                sample.view(num_samples, 1, 28, 28),
                os.path.join(self.config.output_dir, f"sample_{epoch}.png"),
            )

    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Configuration: {self.config}")

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            test_loss = self.test_epoch(epoch)
            self.generate_samples(epoch)


def main():
    """Main function"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Create configuration
    config = Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        no_cuda=args.no_cuda,
        seed=args.seed,
        log_interval=args.log_interval,
        category=args.category,
        alpha=args.alpha,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    try:
        # Setup device and random seeds
        device = setup_device_and_seed(config)

        # Create data loaders
        train_loader, test_loader = create_data_loaders(config)

        # Create model
        model = DirVAE(config, device).to(device)
        print(
            f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )

        # Create trainer and start training
        trainer = DirVAETrainer(model, config, device, train_loader, test_loader)
        trainer.train()

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
