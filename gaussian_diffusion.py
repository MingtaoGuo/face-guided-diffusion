import torch 
import torch.nn.functional as F 
from tqdm import tqdm 
import numpy as np 


class GaussianDiffusion:
    def __init__(self, noise_type, timesteps, device="cpu") -> None:
        self.device = device
        self.timesteps = timesteps
        if noise_type == "linear":
            self.betas = self.linear_beta_schedule(timesteps).to(device)
        else:
            self.betas = self.cos_beta_schedule(timesteps).to(device) 
        self.alphas = 1 - self.betas 
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) 
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], pad=(1, 0), value=1.)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * self.betas
        self.posterior_log_variance = torch.log(self.posterior_variance.clamp(min=1e-15))
        self.posterior_mean_coe1 = torch.sqrt(self.alphas_cumprod_prev) * self.betas / (1 - self.alphas_cumprod)
        self.posterior_mean_coe2 = torch.sqrt(self.alphas) * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def p_sample(self, model, x_t, t, y=None):
        with torch.no_grad():
            model_output = model(x_t, t, y)
            pred_noise, pred_frac = torch.split(model_output, 3, dim=1)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(pred_noise, pred_frac, x_t, t, clip_denoised=True)
        eps = torch.randn(1, 3, 32, 32).cuda()
        if t.cpu().numpy() == 0:
            x_t_1 = model_mean
        else:
            x_t_1 = model_mean + (0.5 * model_log_variance).exp() * eps

        return x_t_1

    def p_sample_loop(self, model, y=None, image_size=32):
        x_t = torch.randn(1, 3, image_size, image_size).to(self.device)
        for t in tqdm(range(self.timesteps - 1, -1, -1)):
            t = torch.tensor([t], dtype=torch.int64).to(self.device)
            x_t_1 = self.p_sample(model, x_t, t, y)
            x_t = x_t_1
        return x_t_1

    def q_sample(self, x_0, t, eps):
        # q{x_t | x_0}
        x_0 = torch.tensor(x_0).to(self.device)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * eps
        return x_t

    def q_posterior_mean_variance(self, x_t, x_0, t):
        # Posterior gaussian distribution q{x_t-1 | x_t, x_0}
        variance = self.posterior_variance.gather(-1, t).view(-1, 1, 1, 1)
        log_variance = self.posterior_log_variance.gather(-1, t).view(-1, 1, 1, 1)
        posterior_mean_coe1_t = self.posterior_mean_coe1.gather(-1, t).view(-1, 1, 1, 1)
        posterior_mean_coe2_t = self.posterior_mean_coe2.gather(-1, t).view(-1, 1, 1, 1)
        mean = posterior_mean_coe1_t * x_0 + posterior_mean_coe2_t * x_t

        return mean, variance, log_variance

    def p_mean_variance(self, pred_noise, pred_frac, x_t, t, clip_denoised=False):
        # Gaussian distribution p{x_t-1 | x_t} predicted by Unet
        betas_t = self.betas.gather(-1, t).view(-1, 1, 1, 1)
        max_log_var = torch.log(betas_t).view(-1, 1, 1, 1)
        min_log_var = self.posterior_log_variance.gather(-1, t).view(-1, 1, 1, 1)
        frac = (pred_frac + 1) / 2
        model_log_variance = frac * max_log_var + (1 - frac) * min_log_var
        model_variance = torch.exp(model_log_variance)

        pred_x_0 = self.predict_x_0_from_noise(x_t, t, pred_noise)
        if clip_denoised: # In training phase set False, reverse diffusion phase set True
            pred_x_0 = torch.clamp(pred_x_0, -1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(x_t, pred_x_0, t)
        return model_mean, model_variance, model_log_variance

    def p_losses(self, model, x_0, t, y=None):
        # Hybrid diffusion loss from the paper improved-DDPM
        eps = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, eps)
        model_pred = model(x_t, t, y)
        pred_noise, pred_frac = torch.split(model_pred, 3, dim=1)
        true_mean, true_variance, true_log_variance = self.q_posterior_mean_variance(x_t, x_0, t)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(pred_noise, pred_frac, x_t, t)
        # L_t-1 = KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        kl = self.gaussian_kl_norm(true_mean, true_log_variance, model_mean.detach(), model_log_variance)
        kl = kl.mean(dim=[1, 2, 3]) / np.log(2)
        # L_0 = -log(p{x_0 | x_1})
        decoder_nll = -self.discretized_gaussian_log_likelihood(x_0, means=model_mean.detach(), log_scales=0.5 * model_log_variance)
        decoder_nll = decoder_nll.mean(dim=[1, 2, 3]) / np.log(2.0)
        L_vlb = torch.where((t == 0), decoder_nll, kl).mean()
        
        L_simple = torch.square(pred_noise - eps).mean()
        L_hybrid = L_simple + L_vlb * 0.001
        return L_hybrid, L_simple, L_vlb

    def gaussian_kl_norm(self, mean1, log_var1, mean2, log_var2):
        kl = 0.5 * (-1.0 + log_var2 - log_var1 + torch.exp(log_var1 - log_var2) + ((mean1 - mean2) ** 2) * torch.exp(-log_var2))
        return kl 

    def discretized_gaussian_log_likelihood(self, x, *, means, log_scales):
        def approx_standard_normal_cdf(x):
            return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        return log_probs

    def linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, timesteps)

    def cos_beta_schedule(self, timesteps, s=0.008):
        t = torch.arange(0, timesteps + 1)
        f_t = torch.cos((t/timesteps + s)/(1 + s)*torch.pi * 0.5) ** 2
        alpha_t_bar = f_t / f_t[0]
        betas = 1 - alpha_t_bar[1:] / alpha_t_bar[:-1]
        return torch.clamp(betas, 0, 0.999)

    def predict_x_0_from_noise(self, x_t, t, eps):
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        x_0 =  (x_t -  sqrt_one_minus_alphas_cumprod * eps) / sqrt_alphas_cumprod
        return x_0


class SpaceSampling(GaussianDiffusion):
    def __init__(self, noise_type, timesteps, num_sample, device="cpu") -> None:
        super().__init__(noise_type, timesteps, device)
        self.device = device
        self.num_sample = num_sample
        # To reduce the number of sampling steps from T to K
        # evenly spaced real numbers in sequence 'S' for improving sampleing speed in the paper improved-DDPM
        self.S = torch.linspace(0, self.timesteps - 1, num_sample, dtype=torch.int64).to(device)
        alphas_cumprod_St = self.alphas_cumprod.gather(-1, self.S)
        alphas_cumprod_St_prev = F.pad(alphas_cumprod_St[:-1], pad=(1, 0), value=1.)
        self.betas = 1 - alphas_cumprod_St / alphas_cumprod_St_prev
        self.alphas = 1 - self.betas 
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) 
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], pad=(1, 0), value=1.)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * self.betas

        self.posterior_log_variance = torch.log(self.posterior_variance.clamp(min=1e-15))
        self.posterior_mean_coe1 = torch.sqrt(self.alphas_cumprod_prev) * self.betas / (1 - self.alphas_cumprod)
        self.posterior_mean_coe2 = torch.sqrt(self.alphas) * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def p_fast_sample(self, model, x_t, t, y=None):
        with torch.no_grad():
            s_t = self.S.gather(-1, t)
            model_output = model(x_t, s_t, y)
            pred_noise, pred_frac = torch.split(model_output, 3, dim=1)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(pred_noise, pred_frac, x_t, t, clip_denoised=True)
        eps = torch.randn_like(x_t).cuda()
        if t.cpu().numpy() == 0:
            x_t_1 = model_mean
        else:
            x_t_1 = model_mean + (0.5 * model_log_variance).exp() * eps

        return x_t_1

    def p_fast_sample_loop(self, model, y=None, image_size=32):
        x_t = torch.randn(1, 3, image_size, image_size).to(self.device)
        res = []
        for i in tqdm(range(self.num_sample - 1, -1, -1)):
            t = torch.tensor([i], dtype=torch.int64).to(self.device)
            x_t_1 = self.p_fast_sample(model, x_t, t, y)
            x_t = x_t_1
            if i % 50 == 0:
                res.append(x_t)
        res = torch.cat(res, dim=3)
        return res


    def p_fast_guidance_sample(self, diff_model, class_model, x_t, t, grad_scale, y=None, num_class=10):
        with torch.no_grad():
            s_t = self.S.gather(-1, t)
            model_output = diff_model(x_t, s_t)
            pred_noise, pred_frac = torch.split(model_output, 3, dim=1)
        
        one_hot_y = F.one_hot(y, num_class)
        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_(True)
            logits = class_model(x_t, t)
            prob = torch.softmax(logits, dim=-1)
            selected = torch.log(torch.sum(prob * one_hot_y, dim=-1).clamp(min=1e-15))
            grad_x_t = torch.autograd.grad(selected, x_t)[0]

        model_mean, model_variance, model_log_variance = self.p_mean_variance(pred_noise, pred_frac, x_t, t, clip_denoised=True)
        shifted_model_mean = model_mean + grad_scale * model_variance * grad_x_t
        eps = torch.randn_like(x_t).cuda()
        if t.cpu().numpy() == 0:
            x_t_1 = shifted_model_mean
        else:
            x_t_1 = shifted_model_mean + (0.5 * model_log_variance).exp() * eps

        return x_t_1

    def p_fast_guidance_sample_loop(self, diff_model, class_model,  grad_scale, y=None, image_size=32, num_class=10):
        x_t = torch.randn(1, 3, image_size, image_size).to(self.device)
        res = []
        for i in tqdm(range(self.num_sample - 1, -1, -1)):
            t = torch.tensor([i], dtype=torch.int64).to(self.device)
            x_t_1 = self.p_fast_guidance_sample(diff_model, class_model, x_t, t, grad_scale, y, num_class)
            x_t = x_t_1
            if i % 50 == 0:
                res.append(x_t)
        res = torch.cat(res, dim=3)
        return res



# from models import UNet
# model = UNet(image_size=32, in_channel=3, out_channel=6, num_class=None, model_channel=128, channel_mult=[1, 2, 2, 2], attention_resolutions=[16, 8], num_res_block=3, num_heads=4, dropout=0.3, num_groups=32, device="cpu")
# gau_diff = GaussianDiffusion(noise_type="cos", timesteps=1000)
# x_0 = torch.randn(1, 3, 32, 32)
# t = torch.randint(0, 1000, [1])
# gau_diff.p_losses(model, x_0, t)