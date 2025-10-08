# Dirichlet Variational Auto-Encoder

Example of Dirichlet-Variational Auto-Encoder (Dir-VAE) by PyTorch.  
Dir-VAE is a VAE which using Dirichlet distribution.

Dir-VAE implemented based on this paper  
[Autoencodeing Variational Inference for Topic Model](https://arxiv.org/pdf/1703.01488) which has been accepted to International Conference on Learning Representations 2017(ICLR2017)  
In the original paper, Dir-VAE(Autoencoded Variational Inference For Topic Mode;AVITM) was proposed for document data.  
This repository, on the other hand, modifies the network architecture of Dir-VAE so that it can be used for image data.

Reconstruction after 10 epochs of training(The top is the original image, the bottom is the reconstructed image):

<div>
	<img src='/image/recon_9.png'>
</div>

[VAE Implementation Reference](https://github.com/pytorch/examples/blob/main/vae/main.py)

## How to run

Install the required libraries using the following command.  
※ Install PyTorch first (XXX should match your CUDA version).  
※ My environment is the following **Pytorch==2.8.0+cu129, CUDA==12.9**  

```bash
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
$ pip install -r requirements.txt
```

You can train model by running main.py.

```bash
 $ python dir_vae.py
```

## About latent variables in Dir-VAE following a Dirichlet distribution

The following is the forward function of Dir-VAE.  
Dir-VAE estimates variables that follow a Dirichlet distribution（dir_z） by inputting variables that follow a normal distribution（gauss_z） after Laplace approximation into a softmax function.  
dir_z is a random variable whose sum is 1.

```python:dir_vae.py
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
```

# Dirichlet Variational Auto-Encoder（日本語）

本リポジトリは VAE の潜在変数を表現する確率分布にディリクレ分布を使用したディリクレ VAE の実装例です．  
厳密には，ディリクレ分布に従う変数の代わりに，ソフトマックス関数から出力される変数として使用しています．

以下の論文を参考に実装を行いました  
[Autoencodeing Variational Inference for Topic Model](https://arxiv.org/pdf/1703.01488)  
元論文ではトピックモデルとして提案され，Bag of words 表現の文書データに適用されました．  
本リポジトリの実装は，VAE のネットワーク構造を改変し，画像に対して使用できるようにしたものです．
