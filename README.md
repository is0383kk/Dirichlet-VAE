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

You need to have pytorch >= v0.4.1 and cuda drivers installed

My environment is the following
Pytorch >= 1.5.1
CUDA >= 10.1  

[VAE Implementation Reference](https://github.com/pytorch/examples/blob/main/vae/main.py)  

## About latent variables in Dir-VAE following a Dirichlet distribution  
The following is the forward function of Dir-VAE.  
Dir-VAE estimates variables that follow a Dirichlet distribution（dir_z） by inputting variables that follow a normal distribution（gauss_z） after Laplace approximation into a softmax function.  
dir_z is a random variable whose sum is 1.
```python:dir_vae.py
def forward(self, x):
    mu, logvar = self.encode(x)
    gauss_z = self.reparameterize(mu, logvar) 
    # gause_z is a variable that follows a multivariate normal distribution
    # Inputting gause_z into softmax func yields a random variable that follows a Dirichlet distribution (Softmax func are used in decoder)
    dir_z = F.softmax(gauss_z,dim=1) # This variable follows a Dirichlet distribution
    return self.decode(gauss_z), mu, logvar, gauss_z, dir_z
```

# Dirichlet Variational Auto-Encoder（日本語）
本リポジトリはVAEの潜在変数を表現する確率分布にディリクレ分布を使用したディリクレVAEの実装例です．  
厳密には，ディリクレ分布に従う変数の代わりに，ソフトマックス関数から出力される変数として使用しています．  

以下の論文を参考に実装を行いました  
[Autoencodeing Variational Inference for Topic Model](https://arxiv.org/pdf/1703.01488)  
元論文ではトピックモデルとして提案され，Bag of words表現の文書データに適用されました．  
本リポジトリの実装は，VAEのネットワーク構造を改変し，画像に対して使用できるようにしたものです．
