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
Splitting the Celeba dataset into four category: "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"
## Training classifier model
```
python train_classifier.py --path /Data_2/gmt/Dataset/img_align_celeba/ --image_size 64 --batchsize 16 --num_class 4
```

## Sampling from pre-trained models
Download the face pre-trained models from [GoogleDrive](), and then put the diffusion model into the folder saved_models, the classifier model into the folder saved_models_classifier

#### Sampling without guidance
```
python test_without_guidance.py --image_size 64 --num_sample 250 --diffusion_model ./saved_models/model_40000.pth --device cuda
```
![](https://github.com/MingtaoGuo/face-guided-diffusion/raw/main/resources/no_guidance.png)


