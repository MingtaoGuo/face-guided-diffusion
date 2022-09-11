import torch 
from torch.optim import Adam
from models import UNetEncoder
from PIL import Image 
import numpy as np 
from Dataset import Dataset_cifar, Dataset_celeba, Dataset_celeba_with_label
from gaussian_diffusion import GaussianDiffusion
from torch.utils.data import DataLoader
import argparse 
import os 


def validation(gau_diff, model, dataloader, batchsize, device):
    model.eval()
    total_acc = 0
    count = 0
    for batch_x0, labels in dataloader:
        batch_x0 = batch_x0.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            t = torch.randint(0, 1, [batchsize]).to(device)
            eps = torch.randn_like(batch_x0).to(device)
            batch_xt = gau_diff.q_sample(batch_x0, t, eps)
            pred = model(batch_xt, t)
            prob = torch.softmax(pred, dim=-1)
            labels = torch.tensor(labels, dtype=torch.int32).to(device)
            acc = labels.eq(torch.argmax(prob, dim=-1)).float().mean()
            total_acc += acc.cpu().numpy()
            count += 1
    model.train()
    return total_acc / count
    

def train(image_size, batchsize, epoch, num_class, path, train_path, val_path, resume, device):
    noise_type = "cos"
    timesteps = 1000
    model_channel=128
    num_heads = 4
    num_res_block = 2
    num_head_channel = 64
    channel_mult = [1, 2, 3, 4]
    attention_resolution = [8, 16, 32]
    dropout = 0.1
    learning_rate = 1e-4
    dataset_train = Dataset_celeba_with_label(img_path=path, anno_path=train_path, image_size=image_size)
    dataset_val = Dataset_celeba_with_label(img_path=path, anno_path=val_path, image_size=image_size)
    model = UNetEncoder(image_size=image_size, 
                        in_channel=3, 
                        out_channel=num_class, 
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
    dataloader_train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=8, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batchsize, shuffle=True, drop_last=True)
    gau_diff = GaussianDiffusion(noise_type=noise_type, timesteps=timesteps, device=device)
    if resume is not None:
        ckpt = torch.load(resume)
        model.load_state_dict(ckpt["model"])
        del ckpt
    if not os.path.exists("./saved_models_classifier"):
        os.mkdir("./saved_models_classifier")
    f = open("loss_classifier.txt", "w")
    print(f"image_size: {str(image_size)}\nnoise_type: {noise_type}\ntimesteps: {str(timesteps)}\nmodel_channels: {str(model_channel)}\nnum_heads: {str(num_heads)}\nnum_res_block: {str(num_res_block)}\nnum_head_channel: {str(num_head_channel)}\nchannel_mult: {str(channel_mult)}\nattention_resolution: {str(attention_resolution)}\ndropout: {str(dropout)}\nlearning_rate: {str(learning_rate)}\n")
    loss_fn = torch.nn.CrossEntropyLoss()
    total_itr = 0
    max_acc = -1
    for epoch in range(epoch):
        for batch_x0, labels in dataloader_train:
            batch_x0 = batch_x0.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            t = torch.randint(0, timesteps, [batchsize]).to(device)
            eps = torch.randn_like(batch_x0).to(device)
            batch_xt = gau_diff.q_sample(batch_x0, t, eps)
            pred = model(batch_xt, t)
            loss = loss_fn(pred, labels)
            loss.backward()
            opt.step()
            if total_itr % 5000 == 0:
                val_acc = validation(gau_diff, model, dataloader_val, batchsize, device)
                if val_acc > max_acc:
                    max_acc = val_acc
                    ckpt = {"model": model.state_dict()}
                    torch.save(ckpt, f"./saved_models_classifier/model_{str(max_acc)}.pth") 
            if total_itr % 100 == 0:
                print(f"Iteration: {total_itr}, Loss: {loss.item()}, Max_val_acc: {max_acc}")
                f.write(f"Iteration: {total_itr}, Loss: {loss.item()}, Max_val_acc: {max_acc}\n")
                f.flush()
            total_itr += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/Data_2/gmt/Dataset/img_align_celeba/")
    parser.add_argument("--train_path", type=str, default="./train.txt")
    parser.add_argument("--val_path", type=str, default="./val.txt")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--num_class", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    train(args.image_size, args.batchsize, args.epoch, args.num_class, args.path, args.train_path, args.val_path, args.resume, args.device)
    # python train.py --path /Data_2/gmt/Dataset/img_align_celeba/ --image_size 64 --batchsize 16
