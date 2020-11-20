# Dirichlet Variational Auto-Encoder
Example of Dirichlet-Variational Auto-Encoder (Dir-VAE)
Dir-VAE is a VAE which using Dirichlet distribution

Dir-VAE implemented based on this paper
Autoencodeing Variational Inference for Topic Model:https://arxiv.org/pdf/1703.01488

Reconstruction after 10 epochs of training:
<div>
	<img src='/image/recon_9.png'>
</div>

You need to have pytorch >= v0.4.1 and cuda drivers installed

My environment is the following
Pytorch==1.5.1
CUDA==10.1

# Dirichlet Variational Auto-Encoder（日本語）
本リポジトリはVAEの潜在変数を表現する確率分布にディリクレ分布を使用したディリクレVAEの実装例です

以下の論文を参考に実装を行いました
"Autoencodeing Variational Inference for Topic Model":https://arxiv.org/pdf/1703.01488
元論文ではトピックモデルとして提案されましたが、本リポジトリは画像に対しても使用できるようにしたものです
