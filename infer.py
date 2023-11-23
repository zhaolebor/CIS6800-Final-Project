import argparse
import os
import itertools
import numpy as np
import time
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Torch related
import torch
from torch import nn, optim

# Local modules
from datasets import Edge2Shoe
from models import (ResNetGenerator, UNetGenerator,
                    Encoder, weights_init_normal,
                    reparameterization, loss_KLD,
                    loss_discriminator, loss_generator
                    )


def norm(image):
    """
    Normalize image tensor
    """
    return (image / 255.0 - 0.5) * 2.0


def denorm(tensor):
    """
    Denormalize image tensor
    """
    return ((tensor + 1.0) / 2.0) * 255.0


def interpolate_z(z0, z1, num_frames):
    """
    Linearly interpolate a latent encoding
    """
    zs = []
    for n in range(num_frames):
        ratio = n / float(num_frames - 1)
        z_t = (1 - ratio) * z0 + ratio * z1
        zs.append(z_t.unsqueeze(0))
    zs = torch.cat(zs, dim=0).float()
    return zs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_random', action="store_true",
                        help='whether to use random latent variable as input')
    parser.add_argument('--infer_encoded', action="store_true",
                        help='whether to use encoded latent variable as input')
    parser.add_argument('--infer_video', action="store_true",
                        help='whether to save frames by interpolating latent variable on a random input')
    parser.add_argument('--compute_lpips', action='store_true',
                        help='whether to generate images for computing LPIPS score')
    parser.add_argument('--exp_id', type=str, default='baseline',
                        help='experiment ID')
    parser.add_argument('--epoch_id', type=int, default=19,
                        help='Model epoch ID to load')
    args = parser.parse_args()

    # Training Configurations
    # (You may put your needed configuration here. Please feel free to add more or use argparse. )
    checkpoints_path = os.path.join(args.exp_id, 'checkpoints/')
    if args.infer_random:
        save_dir = os.path.join(args.exp_id, 'out_images_infer/')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.join(args.exp_id, 'out_images_fid/')
        os.makedirs(save_dir + 'gen', exist_ok=True)
        os.makedirs(save_dir + 'real', exist_ok=True)

    if args.compute_lpips:
        lpips_save_dir = os.path.join(args.exp_id, 'out_images_lpips')
        os.makedirs(lpips_save_dir, exist_ok=True)

    img_dir = 'data/edges2shoes/val/'
    img_shape = (3, 128, 128)  # Please use this image dimension faster training purpose
    n_residual_blocks = 6
    num_epochs = 20
    batch_size = 1
    latent_dim = 8             # latent dimension for the encoded images from domain B
    gpu_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Random seeds (optional)
    torch.manual_seed(1)
    np.random.seed(1)

    # Define DataLoader
    dataset = Edge2Shoe(img_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Loss functions
    l1_loss = torch.nn.L1Loss().to(gpu_id)
    mse_loss = torch.nn.MSELoss().to(gpu_id)

    # Define generator, encoder and discriminators
    # generator = ResNetGenerator(latent_dim, img_shape, n_residual_blocks).to(gpu_id)
    generator = UNetGenerator(latent_dim, img_shape).to(gpu_id)
    encoder = Encoder(latent_dim).to(gpu_id)

    path = os.path.join(checkpoints_path, f'bicycleGAN_epoch_{args.epoch_id}')
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    encoder.eval()
    generator.eval()

    if args.infer_video:
        for i, img_idx in enumerate([10, 20, 30, 40]):
            edge_tensor, rgb_tensor = dataset[img_idx]
            edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
            real_A = edge_tensor[None]
            real_B = rgb_tensor[None]

            z_samples = torch.randn(10, latent_dim).to(gpu_id)
            z_samples_interp = torch.cat(
                [interpolate_z(z_samples[i], z_samples[i + 1], 45) for i in range(len(z_samples) - 1)])

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(os.path.join(args.exp_id, f'demo{i}.mp4'), fourcc, 30, (256, 128))
    
            for frame_idx in range(len(z_samples_interp)):
                fake_B_random = generator.forward(real_A, z_samples_interp[frame_idx][None])
                vis_fake_B_random = denorm(fake_B_random[0].detach()).cpu().numpy().astype(np.uint8)
                vis_real_A = denorm(real_A[0].detach()).cpu().numpy().astype(np.uint8)
                frame = np.concatenate((vis_real_A.transpose(1, 2, 0), vis_fake_B_random.transpose(1, 2, 0)), axis=1)

                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

    for idx, data in tqdm(enumerate(loader)):
        # ######## Process Inputs ##########
        edge_tensor, rgb_tensor = data
        edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
        real_A = edge_tensor
        real_B = rgb_tensor

        if args.infer_random:
            """
            random latent -> B
            """
            fig, axs = plt.subplots(1, 5, figsize=(10, 2))
            vis_real_A = denorm(real_A[0].detach()).cpu().numpy().astype(np.uint8)
            axs[0].imshow(vis_real_A.transpose(1, 2, 0))
            axs[0].set_title('real images')

            z_samples = torch.randn(4, real_A.shape[0], latent_dim).to(gpu_id)
            for i in range(4):
                fake_B_random = generator.forward(real_A, z_samples[i])

                # -------------------------------
                #  Visualization
                # ------------------------------
                vis_fake_B_random = denorm(fake_B_random[0].detach()).cpu().numpy().astype(np.uint8)

                axs[i + 1].imshow(vis_fake_B_random.transpose(1, 2, 0))

            path = os.path.join(save_dir, f'epoch_{args.epoch_id}_{idx}.png')
            plt.savefig(path)
            plt.close()
        elif args.infer_encoded:
            """
            B -> encoded latent -> B
            """
            z_mu, z_logvar = encoder.forward(rgb_tensor)
            z_encoded = reparameterization(z_mu, z_logvar)

            fake_B_encoded = generator.forward(real_A, z_encoded)

            for i in range(batch_size):
                vis_fake_B_encoded = \
                    denorm(fake_B_encoded[i].detach()).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
                path = os.path.join(save_dir, f'gen/fake_B_{idx}_{i}.png')
                plt.imsave(path, vis_fake_B_encoded)
                path = os.path.join(save_dir, f'real/real_B_{idx}_{i}.png')
                plt.imsave(path,
                           denorm(real_B[i].detach()).cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
        elif args.compute_lpips:
            lpips_batch_path = os.path.join(lpips_save_dir, str(idx))
            os.makedirs(lpips_batch_path, exist_ok=True)

            z_samples = torch.randn(10, real_A.shape[0], latent_dim).to(gpu_id)
            for i in range(10):
                fake_B_random = generator.forward(real_A, z_samples[i])

                # -------------------------------
                #  Visualization and Save
                # ------------------------------
                vis_fake_B_random = denorm(fake_B_random[0].detach()).cpu().numpy().astype(np.uint8)

                path = os.path.join(lpips_batch_path, f'img_{idx}_{i}.png')
                plt.imsave(path, vis_fake_B_random.transpose(1, 2, 0))