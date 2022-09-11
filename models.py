from curses.ascii import isupper
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
from abc import abstractmethod

class ResBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, ResBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class TimestepEmbedding(nn.Module):
    def __init__(self, ch, device) -> None:
        super().__init__()
        self.ch = ch
        self.device = device

    def forward(self, timesteps):
        half_dim = self.ch // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.range(0, half_dim - 1) * -emb).to(self.device)
        emb = timesteps[:, None] * emb[None, :] # b x 1 * 1 x n
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
        
        return emb

class DownSample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.avg_pool(x)
        return x

class UpSample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nearest_up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        x = self.nearest_up(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, ch_emb, dropout=0.3, num_groups=32, is_down=False, is_up=False) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(nn.GroupNorm(num_groups, ch_in), 
                                       nn.SiLU(),
                                       nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1))
        self.is_up_down = is_down or is_up
        if is_up:
            self.h_upd = UpSample()
            self.x_upd = UpSample()
        elif is_down:
            self.h_upd = DownSample()
            self.x_upd = DownSample()
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(nn.SiLU(),
                                        nn.Linear(ch_emb, 2*ch_out))
        self.norm = nn.GroupNorm(num_groups, ch_out)
        self.out_layers = nn.Sequential(nn.SiLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1))
        if ch_in == ch_out:
            self.identity = nn.Identity()
        else:
            self.identity = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        if self.is_up_down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(temb)
        scale, shift = torch.chunk(emb_out[..., None, None], 2, dim=1)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.identity(x) + h


class MultiHeadAttention(nn.Module):
    def __init__(self, ch_in, num_heads=4, num_head_channels=64, num_groups=32) -> None:
        super().__init__()
        self.ch_in = ch_in
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups, ch_in)
        self.heads = nn.ModuleList([])
        for i in range(num_heads):
            q = nn.Conv2d(ch_in, num_head_channels, kernel_size=1, stride=1, padding=0)
            k = nn.Conv2d(ch_in, num_head_channels, kernel_size=1, stride=1, padding=0)
            v = nn.Conv2d(ch_in, num_head_channels, kernel_size=1, stride=1, padding=0)
            self.heads.append(nn.ModuleList([q, k, v]))
        self.proj = nn.Conv2d(num_head_channels * num_heads, ch_in, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        h = self.norm(x)
        heads = []
        for f_q, f_k, f_v in self.heads:
            q, k, v = f_q(h), f_k(h), f_v(h)
            heads.append(self.attention(q, k, v))
        heads = torch.concat(heads, dim=1)
        heads = self.proj(heads)
        return heads + x

    def attention(self, q, k, v):
        [b, c, h, w] = q.shape
        attn_score = torch.bmm(q.view(b, c, h*w).permute(0, 2, 1), k.view(b, c, h*w)) / math.sqrt(c)
        attn_weights = torch.softmax(attn_score, dim=-1)
        out = torch.bmm(attn_weights, v.view(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).view(b, c, h, w)
        return out 


class UNet(nn.Module):
    def __init__(self, image_size=32, in_channel=3, out_channel=3, num_head_channel=64, num_class=None, model_channel=128, channel_mult=[1, 2, 4, 8], attention_resolutions=[16, 8], num_res_block=2, num_heads=4, dropout=0.3, num_groups=32, device="cuda") -> None:
        super().__init__()
        self.timestep_embedding = nn.Sequential(TimestepEmbedding(model_channel, device),
                                                nn.Linear(model_channel, model_channel*4),
                                                nn.SiLU(),
                                                nn.Linear(model_channel*4, model_channel*4))
        self.num_class = num_class
        if num_class != None:
            self.label_embedding = nn.Linear(num_class, model_channel*4)
        input_block_chans = [model_channel]
        self.input_block = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channel, model_channel, kernel_size=3, stride=1, padding=1))])
        ch = model_channel
        for idx, mult in enumerate(channel_mult):
            for j in range(num_res_block):
                layers = [ResBlock(ch, model_channel * mult, ch_emb=model_channel*4, dropout=dropout, num_groups=num_groups)]
                ch = mult * model_channel
                resolution = image_size//(2**idx)
                if resolution in attention_resolutions:
                    layers.append(MultiHeadAttention(ch, num_heads=num_heads, num_head_channels=num_head_channel, num_groups=num_groups))
                self.input_block.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if idx < len(channel_mult) - 1:
                self.input_block.append(TimestepEmbedSequential(ResBlock(ch, ch, ch_emb=model_channel*4, dropout=dropout, num_groups=num_groups, is_down=True)))
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(
                             ResBlock(ch, ch, ch_emb=model_channel*4, dropout=dropout, num_groups=num_groups),
                             MultiHeadAttention(ch, num_head_channels=num_head_channel, num_heads=num_heads, num_groups=num_groups),
                             ResBlock(ch, ch, ch_emb=model_channel*4, dropout=dropout, num_groups=num_groups)
                             )
        self.output_block = nn.ModuleList([])
        c = 0
        for idx, mult in list(enumerate(channel_mult))[::-1]:
            c += 1
            for i in range(num_res_block + 1):
                layers = []
                layers.append(ResBlock(ch + input_block_chans.pop(), model_channel * mult, ch_emb=model_channel*4, dropout=dropout, num_groups=num_groups))
                ch = model_channel * mult
                if resolution * 2 ** (c-1) in attention_resolutions:
                    layers.append(MultiHeadAttention(ch, num_head_channels=num_head_channel, num_heads=num_heads, num_groups=num_groups))
                if idx > 0 and i == num_res_block:
                     layers.append(ResBlock(ch, ch, ch_emb=model_channel*4, dropout=dropout, num_groups=num_groups, is_up=True))
                self.output_block.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(nn.GroupNorm(num_groups=num_groups, num_channels=ch), 
                                 nn.SiLU(),
                                 nn.Conv2d(ch, out_channel, kernel_size=3, stride=1, padding=1))

    def forward(self, x, t, y=None):
        temb = self.timestep_embedding(t)
        if self.num_class != None:
            temb = self.label_embedding(y) + temb
        hs = []
        for block in self.input_block:
            x = block(x, temb)
            hs.append(x)
        x = self.middle_block(x, temb)
        for block in self.output_block:
            x = torch.cat([x, hs.pop()], dim=1)
            x = block(x, temb)
        x = self.out(x)

        return x 


class UNetEncoder(nn.Module):
    def __init__(self, image_size=32, in_channel=3, out_channel=3, num_head_channel=64, model_channel=128, channel_mult=[1, 2, 4, 8], attention_resolutions=[16, 8], num_res_block=2, num_heads=4, dropout=0.3, num_groups=32, device="cuda") -> None:
        super().__init__()
        self.timestep_embedding = nn.Sequential(TimestepEmbedding(model_channel, device),
                                                nn.Linear(model_channel, model_channel*4),
                                                nn.SiLU(),
                                                nn.Linear(model_channel*4, model_channel*4))
        input_block_chans = [model_channel]
        self.input_block = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channel, model_channel, kernel_size=3, stride=1, padding=1))])
        ch = model_channel
        for idx, mult in enumerate(channel_mult):
            for j in range(num_res_block):
                layers = [ResBlock(ch, model_channel * mult, ch_emb=model_channel*4, dropout=dropout, num_groups=num_groups)]
                ch = mult * model_channel
                resolution = image_size//(2**idx)
                if resolution in attention_resolutions:
                    layers.append(MultiHeadAttention(ch, num_heads=num_heads, num_head_channels=num_head_channel, num_groups=num_groups))
                self.input_block.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if idx < len(channel_mult) - 1:
                self.input_block.append(TimestepEmbedSequential(ResBlock(ch, ch, ch_emb=model_channel*4, dropout=dropout, num_groups=num_groups, is_down=True)))
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(
                                 ResBlock(ch, ch, ch_emb=model_channel*4, dropout=dropout, num_groups=num_groups),
                                 MultiHeadAttention(ch, num_head_channels=num_head_channel, num_heads=num_heads, num_groups=num_groups),
                                 ResBlock(ch, ch, ch_emb=model_channel*4, dropout=dropout, num_groups=num_groups))
        self.out = nn.Sequential(nn.GroupNorm(num_groups, ch),
                                 nn.SiLU(),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Conv2d(ch, out_channel, kernel_size=1, stride=1, padding=0),
                                 nn.Flatten())
        

    def forward(self, x, t):
        temb = self.timestep_embedding(t)
        for block in self.input_block:
            x = block(x, temb)
        x = self.middle_block(x, temb)
        x = self.out(x)

        return x 

# model = UNet().cuda()
# x = torch.randn(1, 3, 32, 32).cuda()
# t = torch.tensor([4]).cuda()
# y = model(x, t)
# a = 0