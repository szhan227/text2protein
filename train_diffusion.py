import torch
from dataset import ProteinDataset, PaddingCollate
from score_sde_pytorch.models.ncsnpp import NCSNpp, UNetModel
from easydict import EasyDict
import yaml
from model.diffusion_sampler import DiffusionSampler


def train(config_path, save_path):

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    min_res_num = config.data.min_res_num
    max_res_num = config.data.max_res_num
    data_path = config.data.dataset_path

    train_dataset = ProteinDataset(data_path, min_res_num=min_res_num, max_res_num=max_res_num, ss_constraints=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=PaddingCollate(max_res_num)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet = UNetModel(config).to(device)
    llm = None # TODO: load Vicuna

    beta_min, beta_max = config.model.beta_min, config.model.beta_max
    betas = torch.linspace(beta_min, beta_max, config.model.num_scales)
    sampler = DiffusionSampler(model=unet,
                               timesteps=config.model.num_scales,
                               betas=betas)

    optimizer = torch.optim.Adam(unet.parameters(),
                                 lr=config.optim.lr,
                                 eps=config.optim.eps,
                                 weight_decay=config.optim.weight_decay
                                 )


    epoch = 0
    num_batch = len(train_loader)
    while True:
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x_start = batch['coords_6d'].to(device)
            B = x_start.shape[0]
            timesteps = torch.randint(0, 1000, (B, )).to(device)
            textual_emb = torch.randn(B, 1, 128).to(device) # TODO change to real text embedding by Vicuna
            # loss = sampler.p_loss(x_start=x_start, t=timesteps, context=textual_emb)
            loss = sampler(x_start=x_start, t=timesteps, context=textual_emb)
            print(f'\r[Epoch {epoch}] [{i + 1}/{num_batch}] [ Loss {loss.item()}]', end='')
            loss.backward()
            optimizer.step()

            if i > 0 and i % 20000 == 0:
                torch.save(unet.state_dict(), f'{save_path}/unet_ep{epoch}_it{i}.pt')
        print()
        epoch += 1


if __name__ == '__main__':
    train('configs/test_config_large.yml', './checkpoints')

