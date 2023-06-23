import torch
from dataset import ProteinDataset, PaddingCollate

if __name__ == '__main__':

    train_ds = ProteinDataset('./pdbs', max_res_num=1000, ss_constraints=False)
    # for sample in protein_set:
    #     for key, val in sample.items():
    #         if hasattr(val, 'shape'):
    #             print(key, val.shape)
    #         else:
    #             print(key, val)
    #     print('-----------------------')
    train_sampler = torch.utils.data.RandomSampler(
        train_ds,
        replacement=True,
        num_samples=2000 * 2
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=2,
        collate_fn=PaddingCollate(1000)
    )
    for item in train_dl:
        for key, val in item.items():
            if hasattr(val, 'shape'):
                print(key, val.shape)
            else:
                print(key, val)
        print('-----------------------')
        break

