import torch
import torch.nn.functional as F
import math


class TimeEmbedding(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb


class DownSampleBlock(torch.nn.Module):
    pass


class MidBlock(torch.nn.Module):
    pass


class UpSampleBlock(torch.nn.Module):
    pass


class UNetBlock(torch.nn.Module):

    def __init__(self, in_channel, out_channel, time_emb_dim, place='down'):
        super().__init__()

        assert place in ['down', 'up', 'mid']

        self.time_mlp = torch.nn.Linear(time_emb_dim, out_channel)

        if place == 'down':
            self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
            self.feedforward = DownSampleBlock()
        elif place == 'up':
            self.conv1 = torch.nn.Conv2d(2 * in_channel, out_channel, kernel_size=3, padding=1)
            self.feedforward = UpSampleBlock()
        elif place == 'mid':
            self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
            self.feedforward = MidBlock()

        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        self.layer_norm1 = torch.nn.LayerNorm(out_channel)

        self.layer_norm2 = torch.nn.LayerNorm(out_channel)

        self.relu = torch.nn.ReLU()

        def forward(self, x, t):
            h = self.layer_norm1(self.relu(self.conv1(x)))
            time_emb = self.time_mlp(t)
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            h = h + time_emb
            h = self.layer_norm2(self.relu(self.conv2(h)))
            h = self.feedforward(h)
            return h




class UNetModel(torch.nn.Module):

    def __init__(self,
                 # in_channels,
                 # model_channels,
                 # out_channels,
                 # num_res_blocks,
                 # attention_resolutions,
                 dropout=0.,
                 channel_mult=(1, 2, 4, 8),
                 conv_resample=True,
                 dims=3,
                 num_classes=None,
                 ):
        super().__init__()

        in_channel = 5
        down_channels = (64, 128, 256, 512)
        up_channels = (512, 256, 128, 64)
        out_channel = 5
        time_emb_dim = 32

        #Time embedding
        self.time_mlp = torch.nn.Sequential()



    def forward(self, x, cond, t):
        '''
        :param x: batched noisy input
        :param cond: textual description embedding
        :param t: a given batched time step (B,)
        :return: batched less noisy output
        '''
        return x