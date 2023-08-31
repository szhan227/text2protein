import torch
from pathlib import Path
import os
from torch.nn import functional as F
from dataset import PaddingCollate
import pickle as pkl
import yaml


def coord_compare():

    sampled_dir = './sampling/coords_6d/test_config/2023_08_15__04_04_10/test'
    ori_path = './../processed-all-pdb-dicts'

    max_length = 256
    collate_fn = PaddingCollate(max_length)

    sampled_file_paths = os.listdir(sampled_dir)
    prefix_len, suffix_len = len('sampled_'), len('.pkl')

    losses = list()
    loss_sum = 0.0
    for sampled_file_name in sampled_file_paths:
        pdb_name = sampled_file_name[prefix_len:-suffix_len]
        gt_file_path = os.path.join(ori_path, pdb_name + '.pt')
        gt_dict = torch.load(gt_file_path)
        gt_coords_6d = gt_dict['coords_6d']
        num_res = gt_coords_6d.shape[1]

        with open(os.path.join(sampled_dir, sampled_file_name), 'rb') as f:
            sampled_coords = pkl.load(f)

        loss = F.mse_loss(gt_coords_6d, sampled_coords[:, :num_res, :num_res]).item()
        losses.append(f'{pdb_name}: {loss}')
        loss_sum += loss

    avg_loss = loss_sum / len(sampled_file_paths)
    losses.append(f'avg_loss: {avg_loss}')

    dump_path = Path(sampled_dir).parent
    with open(dump_path.joinpath('losses.txt'), 'w') as ff:
        yaml.dump(losses, ff)


if __name__ == '__main__':
    coord_compare()


