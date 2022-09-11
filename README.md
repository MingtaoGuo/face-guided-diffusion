# face-guided-diffusion
Diffusion Models Beat GANs on Image Synthesis (guided diffusion)

## Introduction
--------------

Unofficial guided diffusion implemented by Mingtao Guo

Paper: [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)

## Training diffusion model
```
git clone https://github.com/MingtaoGuo/face-guided-diffusion.git
cd face-guided-diffusion
python train_diffusion.py --path ./img_align_celeba/ --image_size 64 --batchsize 16
```
## Training classifier model
```
python train_classifier.py --path /Data_2/gmt/Dataset/img_align_celeba/ --image_size 64 --batchsize 16 --num_class 4
```
![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/diffusion1000.png)
## Reverse diffusion process
Download the cifar10 and celeba pretrained model from GoogleDrive [cifar10](https://drive.google.com/file/d/1-fFUkAsGi1uHQxWXmkHtt7LwnDzm7odN/view?usp=sharing), [celeba](https://drive.google.com/file/d/17BIK3x-hSfPycFkxW-QxwvUKN0G-b4fv/view?usp=sharing), and then put the model into the folder saved_models
```
python reverse_diffusion_process.py --data_type cifar10 --timesteps 1000 --weights ./saved_models/model500.pth
```
![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/rev_diff.png)

```
python reverse_diffusion_process.py --data_type celeba --timesteps 1000 --weights ./saved_models/model20.pth
```
![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/celeba_rev_diff.png)

## Interpolation
```
python interpolate.py --data_type celeba  --timesteps 1000 --weights ./saved_models/model20.pth --interp_step 500 --img1_path resources/000001.jpg --img2_path resources/000002.jpg
```
![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/celeba_rev_diff_interp.png)
