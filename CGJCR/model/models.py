import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import orthogonal_
from torch.nn.utils import spectral_norm
import torchvision.models as models
from stylegan2.model1 import Generator
def weights_init(m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
				nn.init.normal_(m.weight.data, 0.0, 0.02)
		elif classname.find('BatchNorm') != -1:
				nn.init.normal_(m.weight.data, 1.0, 0.02)
				nn.init.constant_(m.bias.data, 0)

class Encoder(nn.Module):
	def __init__(self, pretrained_weights_path=None):
		super(Encoder, self).__init__()

		# 创建 ResNet-50 模型
		resnet50 = models.resnet50()

		# 如果提供了预训练权重文件路径，则加载权重
		if pretrained_weights_path is not None:
			state_dict = torch.load(pretrained_weights_path)
			resnet50.load_state_dict(state_dict)

		# 去掉 ResNet-50 最后的全连接层
		self.resnet_layers = nn.Sequential(*list(resnet50.children())[:-1])

		# 添加自定义的线性层
		self.fc_mean = nn.Linear(2048, 512)
		self.fc_logvar = nn.Linear(2048, 512)

	def forward(self, x):
		# 提取特征
		features = self.resnet_layers(x)
		features = features.view(features.size(0), -1)
		#features = features.unsqueeze(-1).unsqueeze(-1)
		# 经过自定义的线性层
		mean = self.fc_mean(features)
		logvar = self.fc_logvar(features)

		return mean, logvar

class Decode(nn.Module):
		def __init__(self, G):
			super(Decode, self).__init__()
			self.generator = G

		def forward(self, z,w_direction,z_direction,w_space):
			#return self.generator(z, None, truncation_psi=0.7, noise_mode='const')

			return self.generator(z, None,w_direction,z_direction,w_space)

def init_weights(m):
		if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
			orthogonal_(m.weight)
			m.bias.data.fill_(0.)


class Self_Attn(nn.Module):
	""" Self attention Layer"""

	def __init__(self, in_channels):
		super().__init__()
		self.in_channels = in_channels
		self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1,
																		padding=0)
		self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1,
																	padding=0)
		self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1,
																padding=0)
		self.snconv1x1_attn = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1, stride=1,
																	 padding=0)
		self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
		self.softmax = nn.Softmax(dim=-1)
		self.sigma = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		"""
				inputs :
						x : input feature maps(B X C X W X H)
				returns :
						out : self attention value + input feature
		"""
		_, ch, h, w = x.size()
		# Theta path
		theta = self.snconv1x1_theta(x)
		theta = theta.view(-1, ch // 8, h * w)
		# Phi path
		phi = self.snconv1x1_phi(x)
		phi = self.maxpool(phi)
		phi = phi.view(-1, ch // 8, h * w // 4)
		# Attn map
		attn = torch.bmm(theta.permute(0, 2, 1), phi)
		attn = self.softmax(attn)
		# g path
		g = self.snconv1x1_g(x)
		g = self.maxpool(g)
		g = g.view(-1, ch // 2, h * w // 4)
		# Attn_g
		attn_g = torch.bmm(g, attn.permute(0, 2, 1))
		attn_g = attn_g.view(-1, ch // 2, h, w)
		attn_g = self.snconv1x1_attn(attn_g)
		# Out
		out = x + self.sigma * attn_g
		return out


def snlinear(in_features, out_features):
	return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
	return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
																 stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


class DiscBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv_1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
		self.conv_2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
		self.conv_0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

		self.relu = nn.LeakyReLU(negative_slope=0.1)
		self.downsample = nn.AvgPool2d(2)
		self.ch_mismatch = False
		if in_channels != out_channels:
			self.ch_mismatch = True

	def forward(self, x, downsample=True):
		x0 = x

		x = self.relu(x)
		x = self.conv_1(x)
		x = self.relu(x)
		x = self.conv_2(x)
		if downsample:
			x = self.downsample(x)

		if downsample or self.ch_mismatch:
			x0 = self.conv_0(x0)
			if downsample:
				x0 = self.downsample(x0)

		out = x + x0
		return out


class Discriminators(nn.Module):
	"""Discriminator."""

	def __init__(self, conv_dim, image_size=128, in_channels=3, out_channels=1, out_feature=False):
		super().__init__()
		self.conv_dim = conv_dim
		self.image_size = image_size
		self.out_feature = out_feature

		self.fromRGB = snconv2d(in_channels, conv_dim, 1, bias=True)

		self.block1 = DiscBlock(conv_dim, conv_dim * 2)
		self.self_attn = Self_Attn(conv_dim * 2)
		self.block2 = DiscBlock(conv_dim * 2, conv_dim * 4)
		self.block3 = DiscBlock(conv_dim * 4, conv_dim * 8)
		if image_size == 64:
			self.block4 = DiscBlock(conv_dim * 8, conv_dim * 16)
			self.block5 = DiscBlock(conv_dim * 16, conv_dim * 16)
		elif image_size == 128:
			self.block4 = DiscBlock(conv_dim * 8, conv_dim * 16)
			self.block5 = DiscBlock(conv_dim * 16, conv_dim * 16)
			self.block6 = DiscBlock(conv_dim * 16, conv_dim * 16)
		else:
			self.block4 = DiscBlock(conv_dim * 8, conv_dim * 8)
			self.block5 = DiscBlock(conv_dim * 8, conv_dim * 16)
			self.block6 = DiscBlock(conv_dim * 16, conv_dim * 16)
		self.relu = nn.LeakyReLU(negative_slope=0.1)
		self.snlinear1 = snlinear(in_features=conv_dim * 16, out_features=1024)
		self.snlinear2 = snlinear(in_features=1024, out_features=out_channels)
		# Weight init
		self.apply(init_weights)
		self.relu = nn.ReLU()

	def forward(self, x):
		h0 = self.fromRGB(x)
		h1 = self.block1(h0)
		h1 = self.self_attn(h1)
		h2 = self.block2(h1)
		h3 = self.block3(h2)
		h4 = self.block4(h3)
		if self.image_size == 64:
			h5 = self.block5(h4, downsample=False)
			h6 = h5
		elif self.image_size == 128:
			h5 = self.block5(h4)
			h6 = self.block6(h5, downsample=False)
		else:
			h5 = self.block5(h4)
			h6 = self.block6(h5)
			h6 = self.block7(h6, downsample=False)
		h6 = self.relu(h6)

		# Global sum pooling
		h7 = torch.sum(h6, dim=[2, 3])
		out = self.relu(self.snlinear1(h7))
		#out = torch.sigmoid(self.snlinear2(out))
		t=self.snlinear2(out)
		out=torch.sigmoid(t)
		if self.out_feature:
			return out, h7
		else:
			return out


class VAE_GAN(nn.Module):
	def __init__(self):
		super(VAE_GAN,self).__init__()
		self.encoder=Encoder("")
		self.decoder=Decoder()
		self.discriminator=Discriminators(conv_dim=32)
		# self.encoder.apply(weights_init)
		self.decoder.apply(weights_init)
		# self.discriminator.apply(weights_init)


	def forward(self,x):
		bs=x.size()[0]
		z_mean,z_logvar=self.encoder(x)
		std = z_logvar.mul(0.5).exp()

		#sampling epsilon from normal distribution
		epsilon=Variable(torch.randn(bs,512)).to('cuda')
		z=z_mean+std*epsilon
		x_tilda=self.decoder(z)

		return z_mean,z_logvar,x_tilda

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

class VAE_sty(nn.Module):
	def __init__(self):
		super(VAE_sty, self).__init__()
		self.encoder = Encoder("")
		G = Generator(log_resolution=7, d_latent=512)
		self.decoder = Decode(G)
		self.discriminator = Discriminators(conv_dim=32)


	def forward(self, x,w_direction,z_direction,w_space):
		bs = x.size()[0]
		z_mean, z_logvar = self.encoder(x)

		std = z_logvar.mul(0.5).exp()

		# sampling epsilon from normal distribution
		epsilon = torch.randn(bs, 512).to(x.device)
		z = z_mean + std * epsilon


		#print(z.min(),z.max())
		x_tilda,w = self.decoder(z,w_direction,z_direction,w_space)

		return z_mean, z_logvar, x_tilda,w,z

