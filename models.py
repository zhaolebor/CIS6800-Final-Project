from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#        Encoder
##############################
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        """
        The encoder used in both cVAE-GAN and cLR-GAN, which encode img B or B_hat to latent vector
        This encoder uses resnet-18 to extract features, and further encode them into a distribution
        similar to VAE encoder.
        Note: You may either add "reparametrization trick" and "KL divergence" in the train.py file
        Args in constructor:
            latent_dim: latent dimension for z
        Args in forward function:
            img: image input (from domain B)
        Returns:
            mu: mean of the latent code
            logvar: sigma of the latent code
        """
        super(Encoder, self).__init__()

        # Extracts features at the last fully-connected
        resnet18_model = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


##############################
#        Generator
##############################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape, n_residual_blocks):
        super().__init__()

        channels = img_shape[0]

        # Initial conv block
        out_channels = 64
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels + latent_dim, out_channels, 7),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        in_channels = out_channels

        # Downsample
        for _ in range(2):
            out_channels *= 2
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # Residual blocks
        for _ in range(n_residual_blocks):
            layers += [ResidualBlock(out_channels)]

        # Upsample
        for _ in range(2):
            out_channels //= 2
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # Final output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(out_channels, channels, 7), nn.Tanh()]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        # reference: https://github.com/junyanz/BicycleGAN/blob/master/models/networks.py
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
            z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat([x, z_img], dim=1)
        return self.layers(x_with_z)


class DownsampleBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class UNetGenerator(nn.Module):
    """
    UNet generator used in both cVAE-GAN and cLR-GAN, which transform A to B
    Args in constructor:
        latent_dim: latent dimension for z
        image_shape: (channel, h, w), you may need this to specify the output dimension (optional)
    Args in forward function:
        x: image input (from domain A)
        z: latent vector (encoded B)
    Returns:
        fake_B: generated image in domain B
    """
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        channels, self.h, self.w = img_shape

        self.down1 = DownsampleBlock(channels + latent_dim, 64, normalize=False)
        self.down2 = DownsampleBlock(64, 128)
        self.down3 = DownsampleBlock(128, 256)
        self.down4 = DownsampleBlock(256, 512)
        self.down5 = DownsampleBlock(512, 512)
        self.down6 = DownsampleBlock(512, 512)
        self.down7 = DownsampleBlock(512, 512, normalize=False)
        self.up1 = UpsampleBlock(512, 512)
        self.up2 = UpsampleBlock(1024, 512)
        self.up3 = UpsampleBlock(1024, 512)
        self.up4 = UpsampleBlock(1024, 256)
        self.up5 = UpsampleBlock(512, 128)
        self.up6 = UpsampleBlock(256, 64)

        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(128, channels, 4, stride=2, padding=1), nn.Tanh()
        )

    def forward(self, x, z):
        # reference: https://github.com/junyanz/BicycleGAN/blob/master/models/networks.py
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
            z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat([x, z_img], dim=1)

        d1 = self.down1(x_with_z)               # d1:(N,64,64,64)
        d2 = self.down2(d1)                     # d2:(N,128,32,32)
        d3 = self.down3(d2)                     # d3:(N,256,16,16)
        d4 = self.down4(d3)                     # d4:(N,512,8,8)
        d5 = self.down5(d4)                     # d5:(N,512,4,4)
        d6 = self.down6(d5)                     # d6:(N,512,2,2)
        d7 = self.down7(d6)                     # d7:(N,512,1,1)
        u1 = self.up1(d7, d6)                   # u1:(N,1024,2,2)
        u2 = self.up2(u1, d5)                   # u2:(N,1024,4,4)
        u3 = self.up3(u2, d4)                   # u3:(N,1024,8,8)
        u4 = self.up4(u3, d3)                   # u4:(N,512,16,16)
        u5 = self.up5(u4, d2)                   # u5:(N,256,32,32)
        u6 = self.up6(u5, d1)                   # u6:(N,128,64,64)

        return self.out_layer(u6)               # out_layer:(N,3,128,128)


##############################
#        Discriminator
##############################
class MultiPatchGANDiscriminator(nn.Module):
    def __init__(self, input_shape, num_scales=2):
        """
        PatchGAN discriminator with input multiple scales
        Args in constructor:
            input_shape: input image dimensions
        Args in forward function:
            x: image input (real_B, fake_B)
        Returns:
            discriminator output: could be a single value or a matrix depending on the type of GAN
        """
        super().__init__()
        self.num_scales = num_scales
        channels, _, _ = input_shape

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.ModuleList()
        for i in range(self.num_scales):
            self.models.add_module(
                f"disc_{i}",
                nn.Sequential(
                    *discriminator_block(channels, 64 // 2**i, normalize=False),
                    *discriminator_block(64 // 2**i, 128 // 2**i),
                    *discriminator_block(128 // 2**i, 256 // 2**i),
                    *discriminator_block(256 // 2**i, 512 // 2**i),
                    nn.ZeroPad2d((1, 0, 1, 0)),
                    nn.Conv2d(512 // 2**i, 1, 4, padding=1)
                )
            )

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, img):
        outputs = []
        for model in self.models:
            outputs.append(model(img))
            img = self.downsample(img)
        return outputs


def reparameterization(mean, log_var, eps=None):
    ################################
    # std = exp(0.5*log_var)
    # eps = N(0,1)
    # z = mean + std * eps
    ################################

    std = torch.exp(0.5 * log_var)
    if eps is None:
        eps = torch.randn_like(mean)
    z = mean + std * eps

    return z


def loss_discriminator(D, fake_B, real_B, valid, fake, criterion):
    """
    The discriminator loss has two parts: MSE(D(real_B), valid) and MSE(D(G(real_A)), fake).
    """
    real_outs = D.forward(real_B)
    fake_outs = D.forward(fake_B.detach())

    loss_real = sum(criterion(out, torch.full_like(out, valid)) for out in real_outs)
    loss_fake = sum(criterion(out, torch.full_like(out, fake)) for out in fake_outs) 
    
    # sum real_B loss and fake loss as the loss_D
    loss_D = loss_real + loss_fake

    return loss_D


def loss_generator(D, fake_B, valid, criterion):
    """
    The generator loss is MSE(D(G(real_A)), valid).
    """
    fake_outs = D.forward(fake_B)
    loss_G = sum(criterion(out, torch.full_like(out, valid)) for out in fake_outs)

    return loss_G


def loss_KLD(mu, log_var):
    """
    Compute KL divergence loss
    mu, log_var will be the computed from Encoder outputs
    """
    loss = 0.5 * torch.sum(torch.exp(log_var) + mu ** 2 - 1.0 - log_var)
    return loss


from datasets import Edge2Shoe


if __name__ == "__main__":
    # Define DataLoader
    img_dir = 'data/edges2shoes/train/'
    img_shape = (3, 128, 128)  # Please use this image dimension faster training purpose

    # load dataset
    dataset = Edge2Shoe(img_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    net_discriminator = MultiPatchGANDiscriminator(input_shape=img_shape)

    for data in loader:
        edge_tensor, rgb_tensor = data

        discriminator_out = net_discriminator.forward(edge_tensor)
        print(discriminator_out[0].shape, discriminator_out[1].shape)