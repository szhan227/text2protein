import torch


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


    def forward(self, x, cond, t):
        '''
        :param x: batched noisy input
        :param cond: textual description embedding
        :param t: a given batched time step (B,)
        :return: batched less noisy output
        '''
        return x