import torch
from dataset import ProteinDataset, PaddingCollate
from score_sde_pytorch.models.ncsnpp import NCSNpp, UNetModel
from easydict import EasyDict
import yaml
from model.diffusion_sampler import DiffusionSampler
from model.attention import SpatialTransformer
import json

if __name__ == '__main__':
    train_ds = ProteinDataset('./pdbs', './db.pt', max_res_num=256, ss_constraints=False)

    train_sampler = torch.utils.data.RandomSampler(
        train_ds,
        replacement=True,
        num_samples=2000 * 2
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=1,
        collate_fn=PaddingCollate(256)
    )

    # with open('./configs/test_config.yml', 'r') as f:
    #     config = EasyDict(yaml.safe_load(f))
    # # print(config)
    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # # if context shape is [B, N], then add one dimension in the mid to get [B, 1, N]
    # text_emb = torch.randn(1, 1, 128).to(device)
    # model = UNetModel(config).to(device)
    # print(model)

    batch = next(iter(train_dl))
    coords_6d = batch['coords_6d'].to(device)
    #
    # # coords_6d = torch.randn(1, 5, 64, 64).to(device)
    print('coords_6d', coords_6d.shape)
    print('id:', batch['id'])
    print('caption:', batch['caption'].shape)

    # d = {
    #     'pdb_id': '2ki6',
    #     'caption_emb': torch.randn(1, 1, 128),
    # }
    #
    # torch.save(d, './db.pt')





    # timesteps = torch.randint(0, 1000, (1, )).to(device)
    # output = model(coords_6d, timesteps, text_emb)
    # print(output.shape)
    #
    # sampler = DiffusionSampler(model).to(device)
    # output = sampler.ddim_sample(shape=(1, 5, 64, 64), context=text_emb)
    #
    #
    # print("after sampling:", output.shape)


    # text_emb = torch.randn(2, 256, 1024).to(device)
    # h = torch.randn(2, 512, 16, 16).to(device)
    # ch = 512
    # num_head_channels = -1
    # nums_head = 8
    # dim_head = ch // nums_head
    # trans = SpatialTransformer(in_channels=ch, n_heads=nums_head, d_head=dim_head, context_dim=1024).to(device)
    # h = trans(x=h, context=text_emb)
    # print(h.shape)






