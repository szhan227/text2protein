import torch
from pathlib import Path
import os
from torch.nn import functional as F
# from dataset import PaddingCollate
import pickle as pkl
import yaml
from tqdm import tqdm


def coord_compare():

    # customize your own path
    sampled_dir = './sampling/coords_6d/test_config/2023_08_15__04_04_10/test'
    ori_path = './../processed-all-pdb-dicts'

    max_length = 256
    # collate_fn = PaddingCollate(max_length)

    sampled_file_paths = os.listdir(sampled_dir)
    prefix_len, suffix_len = len('sampled_'), len('.pkl')

    to_save = dict()
    losses = dict()
    loss_sum = 0.0

    progress_bar = tqdm(sampled_file_paths)
    for sampled_file_name in progress_bar:
        pdb_name = sampled_file_name[prefix_len:-suffix_len]
        gt_file_path = os.path.join(ori_path, pdb_name + '.pt')
        gt_dict = torch.load(gt_file_path)
        gt_coords_6d = gt_dict['coords_6d']
        num_res = gt_coords_6d.shape[1]

        with open(os.path.join(sampled_dir, sampled_file_name), 'rb') as f:
            sampled_coords = pkl.load(f)
        if len(sampled_coords.shape) == 4:
            sampled_coords = sampled_coords[0]

        loss = F.mse_loss(gt_coords_6d, sampled_coords[:, :num_res, :num_res]).item()
        # losses.append(f'{pdb_name}: {loss}')
        losses[pdb_name] = loss
        loss_sum += loss
        progress_bar.set_description(f'{pdb_name}: {loss}')

    avg_loss = sum(losses.values()) / len(losses)
    min_loss = min(losses.values())
    max_loss = max(losses.values())
    std_loss = torch.std(torch.tensor(list(losses.values()))).item()
    to_save['losses'] = losses
    to_save['avg_loss'] = avg_loss
    to_save['min_loss'] = min_loss
    to_save['max_loss'] = max_loss
    to_save['std_loss'] = std_loss


    dump_path = Path(sampled_dir).parent
    with open(dump_path.joinpath('coords_6d_losses.yaml'), 'w') as ff:
        yaml.dump(to_save, ff)


if __name__ == '__main__':
    coord_compare()


