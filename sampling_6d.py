import torch
from pathlib import Path
from score_sde_pytorch.utils import get_model, restore_checkpoint, recursive_to
from score_sde_pytorch.models.ema import ExponentialMovingAverage
import score_sde_pytorch.sde_lib as sde_lib
import score_sde_pytorch.sampling as sampling
import score_sde_pytorch.losses as losses
import pickle as pkl
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from utils import get_conditions_random, get_mask_all_lengths, get_conditions_from_pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--pdb', type=str, default=None)
    parser.add_argument('--chain', type=str, default="A")
    parser.add_argument('--mask_info', type=str, default="1:5,10:15")
    parser.add_argument('--tag', type=str, default="test")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--select_length', type=bool, default=False)
    parser.add_argument('--length_index', type=int, default=1) # Index starts at 1
    args = parser.parse_args()

    assert not (args.pdb is not None and args.select_length)

    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    config.device = args.device
    workdir = Path("sampling", "coords_6d", Path(args.config).stem, Path(args.checkpoint).stem, args.tag)

    # Initialize model.
    score_model = get_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    state = restore_checkpoint(args.checkpoint, state, args.device)
    state['ema'].store(state["model"].parameters())
    state['ema'].copy_to(state["model"].parameters())

    # Load SDE
    if config.training.sde == "vesde":
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    elif config.training.sde == "vpsde":
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3

    # Sampling function
    sampling_shape = (args.batch_size, config.data.num_channels,
                      config.data.max_res_num, config.data.max_res_num)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)

    generated_samples = []
    for _ in tqdm(range(args.n_iter)):
        if args.select_length:
            mask = get_mask_all_lengths(config,batch_size=args.batch_size)[args.length_index-1]
            condition = {"length": mask.to(config.device)}
        elif args.pdb is not None:
            condition = get_conditions_from_pdb(args.pdb, config, args.chain, args.mask_info, batch_size=args.batch_size)
        else:
            condition = get_conditions_random(config, batch_size=args.batch_size)
        sample, n = sampling_fn(state["model"], condition)
        generated_samples.append(sample.cpu())

    generated_samples = torch.cat(generated_samples, 0)

    workdir.mkdir(parents=True, exist_ok=True)
    with open(workdir.joinpath("samples.pkl"), "wb") as f:
        pkl.dump(generated_samples, f)

if __name__ == "__main__":
    main()
