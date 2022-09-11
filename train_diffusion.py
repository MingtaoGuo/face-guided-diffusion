import torch 
from torch.optim import Adam
from models import UNet
from PIL import Image 
import numpy as np 
from Dataset import Dataset_cifar, Dataset_celeba
from gaussian_diffusion import GaussianDiffusion, SpaceSampling
from torch.utils.data import DataLoader
from ema_pytorch import EMA
import argparse 
import os 


def train(image_size, batchsize, epoch, num_class, num_sample, path, ema_decay, ema_update_every, resume, device):
    if image_size == 32:
        noise_type = "cos"
        timesteps = 4000
        model_channel=128
        num_heads = 4
        num_res_block = 2
        num_head_channel = 32
        channel_mult = [1, 2, 2, 2]
        attention_resolution = [8, 16]
        dropout = 0.3
        learning_rate = 1e-4
        dataset = Dataset_cifar(path=path)
    elif image_size == 64:
        noise_type = "cos"
        timesteps = 1000
        model_channel=192
        num_heads = 4
        num_res_block = 3
        num_head_channel = 64
        channel_mult = [1, 2, 3, 4]
        attention_resolution = [8, 16, 32]
        dropout = 0.1
        learning_rate = 1e-4
        dataset = Dataset_celeba(path=path, image_size=image_size)
    elif image_size == 128:
        noise_type = "linear"
        timesteps = 1000
        model_channel = 256
        num_heads = 4
        num_res_block = 2
        num_head_channel = 64
        channel_mult = [1, 1, 2, 3, 4]
        attention_resolution = [8, 16, 32]
        dropout = 0.
        learning_rate = 1e-4
        dataset = Dataset_celeba(path=path, image_size=image_size)
    elif image_size == 256:
        noise_type = "linear"
        timesteps = 1000
        model_channel = 256
        num_heads = 4
        num_res_block = 2
        num_head_channel = 64
        channel_mult = [1, 1, 2, 2, 4, 4]
        attention_resolution = [8, 16, 32]
        dropout = 0.
        learning_rate = 1e-4
        dataset = Dataset_celeba(path=path, image_size=image_size)

    model = UNet(image_size=image_size, 
                 in_channel=3, 
                 out_channel=6, # [pred_noise, pred_log_var_frac]
                 num_class=num_class, 
                 model_channel=model_channel, 
                 channel_mult=channel_mult, 
                 attention_resolutions=attention_resolution, 
                 num_res_block=num_res_block, 
                 num_head_channel=num_head_channel, 
                 num_heads=num_heads, 
                 dropout=dropout, 
                 num_groups=32, 
                 device=device)
    model = model.to(device)
    model.train()
    opt = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=8, drop_last=True)
    gau_diff = GaussianDiffusion(noise_type=noise_type, timesteps=timesteps, device=device)
    space_sample = SpaceSampling(noise_type=noise_type, timesteps=timesteps, num_sample=num_sample, device=device)
    ema_model = EMA(model, beta = ema_decay, update_every = ema_update_every)
    ema_model = ema_model.to(device)
    if resume is not None:
        ckpt = torch.load(resume)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        del ckpt
    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./saved_models"):
        os.mkdir("./saved_models")
    f = open("loss.txt", "w")
    print(f"image_size: {str(image_size)}\nnoise_type: {noise_type}\ntimesteps: {str(timesteps)}\nmodel_channels: {str(model_channel)}\nnum_heads: {str(num_heads)}\nnum_res_block: {str(num_res_block)}\nnum_head_channel: {str(num_head_channel)}\nchannel_mult: {str(channel_mult)}\nattention_resolution: {str(attention_resolution)}\ndropout: {str(dropout)}\nlearning_rate: {str(learning_rate)}\n")

    total_itr = 0
    for epoch in range(epoch):
        for itr, [batch_x0, labels] in enumerate(dataloader):
            batch_x0 = batch_x0.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            t = torch.randint(0, timesteps, [batchsize]).to(device)
            loss, L_simple, L_vlb = gau_diff.p_losses(model, batch_x0, t, labels)
            loss.backward()
            opt.step()
            ema_model.update()
            if total_itr % 100 == 0:
                print(f"Iteration: {total_itr}, Loss: {loss.item()}, L_simple: {L_simple.item()}, L_vlb: {L_vlb.item()}")
                f.write(f"Iteration: {total_itr}, Loss: {loss.item()}, L_simple: {L_simple.item()}, L_vlb: {L_vlb.item()}\n")
                f.flush()
            if total_itr % 5000 == 0:
                ckpt = {"model": model.state_dict(), "ema": ema_model.state_dict()}
                torch.save(ckpt, f"./saved_models/model_{str(total_itr)}.pth") 
            if total_itr % 1000 == 0:
                ema_model.eval()
                pred_x_0 = space_sample.p_fast_sample_loop(ema_model, y=None, image_size=image_size)
                pred_x_0_ = ((pred_x_0.permute(0, 2, 3, 1)).clamp(-1, 1) + 1) * 127.5
                pred_x_0_ = pred_x_0_.cpu().numpy()[0]
                Image.fromarray(np.uint8(pred_x_0_)).save(f"./results/{str(total_itr)}.png")
            total_itr += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/Data_2/gmt/Dataset/img_align_celeba/")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--num_class", type=int, default=None)
    parser.add_argument("--num_sample", type=int, default=250)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    train(args.image_size, args.batchsize, args.epoch, args.num_class, args.num_sample, args.path, args.ema_decay, args.ema_update_every, args.resume, args.device)
    # python train_diffusion.py --path /Data_2/gmt/Dataset/img_align_celeba/ --image_size 64 --batchsize 16
