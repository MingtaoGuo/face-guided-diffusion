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
We trained the diffusion model about 200,000 iterations with batchsize 16 two days, and the classifier model about 140,000 iterations one day in a single GTX 3090 24GB.

Download the face pre-trained models from [GoogleDrive](), and then put the diffusion model into the folder saved_models, the classifier model into the folder saved_models_classifier

#### Sampling without guidance
```
python test_without_guidance.py --image_size 64 --num_sample 250 --diffusion_model ./saved_models/model_200000.pth --device cuda
```
|   1  | 2  |   3  | 4  |
|  ----  | ----  |  ----  | ----  |
|![](https://github.com/MingtaoGuo/face-guided-diffusion/raw/main/resources/no_guidance.png)|![](https://github.com/MingtaoGuo/face-guided-diffusion/raw/main/resources/no_guidance1.png)|![](https://github.com/MingtaoGuo/face-guided-diffusion/raw/main/resources/no_guidance2.png)|![](https://github.com/MingtaoGuo/face-guided-diffusion/raw/main/resources/no_guidance3.png)|

#### Sampling with guidance
```
python test_with_guidance.py --image_size 64 --num_class 4 --diffusion_model ./saved_models/model_200000.pth --classifier_model ./saved_models_classifier/model_0.9147636217948718.pth --num_sample 250 --label 0 --grad_scale 5
```
|  Black_Hair   | Blond_Hair  |  Brown_Hair   | Gray_Hair  |
|  ----  | ----  |  ----  | ----  |
| ![](https://github.com/MingtaoGuo/face-guided-diffusion/raw/main/resources/out_0.png) | ![](https://github.com/MingtaoGuo/face-guided-diffusion/raw/main/resources/out_1.png) |![](https://github.com/MingtaoGuo/face-guided-diffusion/raw/main/resources/out_2.png)  | ![](https://github.com/MingtaoGuo/face-guided-diffusion/raw/main/resources/out_3.png) |

## Acknowledgements
* Official pytorch implementation [improved-diffusion](https://github.com/openai/improved-diffusion)
* Official pytorch implementation [guided-diffusion](https://github.com/openai/guided-diffusion)
* Unofficial pytorch implementation [ddpm](https://github.com/lucidrains/denoising-diffusion-pytorch)
## Author 
Mingtao Guo
E-mail: gmt798714378@hotmail.com
