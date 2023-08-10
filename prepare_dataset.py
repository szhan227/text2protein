import torch
from dataset import ProteinDataset, ProteinProcessedDataset, PaddingCollate
from easydict import EasyDict
import yaml


def prepare_dataset():
    config_path = './configs/test_config.yml'
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    device = config.device
    dataset_path = config.data.dataset_path
    caption_path = config.data.caption_path
    ss_constraints = True if config.data.num_channels == 8 else False
    dataset = ProteinDataset(dataset_path=dataset_path,
                             description_path=caption_path,
                             min_res_num=config.data.min_res_num,
                             max_res_num=config.data.max_res_num,
                             ss_constraints=ss_constraints)


if __name__ == '__main__':
    prepare_dataset()