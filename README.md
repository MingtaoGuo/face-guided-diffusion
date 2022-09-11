# face-guided-diffusion
Diffusion Models Beat GANs on Image Synthesis (guided diffusion)

## Introduction
--------------

Unofficial guided diffusion implemented by Mingtao Guo

Paper: [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)

## Training diffusion model
Download the [Celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
```
git clone https://github.com/MingtaoGuo/face-guided-diffusion.git
cd face-guided-diffusion
unzip img_align_celeba.zip
python train_diffusion.py --path ./img_align_celeba/ --image_size 64 --batchsize 16
```
## Training classifier model
```
python train_classifier.py --path /Data_2/gmt/Dataset/img_align_celeba/ --image_size 64 --batchsize 16 --num_class 4
```

## Sampling from pre-trained models
Download the face pre-trained models from [GoogleDrive](), and then put the diffusion model into the folder saved_models, the classifier model into the folder saved_models_classifier
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
