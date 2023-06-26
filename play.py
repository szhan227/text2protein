import torch
from dataset import ProteinDataset, PaddingCollate
from score_sde_pytorch.models.ncsnpp import NCSNpp
from easydict import EasyDict
import yaml
from transformers import LlamaTokenizer

if __name__ == '__main__':
    # train_ds = ProteinDataset('./pdbs', max_res_num=256, ss_constraints=False)
    #
    # train_sampler = torch.utils.data.RandomSampler(
    #     train_ds,
    #     replacement=True,
    #     num_samples=2000 * 2
    # )
    # train_dl = torch.utils.data.DataLoader(
    #     train_ds,
    #     sampler=train_sampler,
    #     batch_size=1,
    #     collate_fn=PaddingCollate(256)
    # )

    with open('./configs/cond_length.yml', 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    # print(config)

    device = 'cuda'
    model = NCSNpp(config).to(device)
    print(model)
    # batch = next(iter(train_dl))
    # coords_6d = batch['coords_6d'].to(device)
    coords_6d = torch.randn(1, 5, 256, 256).to(device)
    print('coords_6d', coords_6d.shape)
    timesteps = torch.randint(0, 1000, (1, )).to(device)
    output = model(coords_6d, timesteps)
    print(output.shape)




