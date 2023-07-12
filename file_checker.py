import os
import json
from pathlib import Path
from tqdm import tqdm
import torch

def compare_pdb_file_and_caption():
    caption_path = './../caption-pdbs/abstract.json'
    pdb_path = './../raw-pdbs'

    caption_name_set = set()
    pdb_name_set = set()

    print('Start to check files')
    with open(caption_path, 'r') as json_file:
        # here json format: key=pdb_id, value=caption_embedding
        ann_json = json.load(json_file)
    print('Load json file done')
    for ann in tqdm(ann_json):
        caption_name_set.add(ann['pdb_id'])

    for root, dirs, files in tqdm(os.walk(pdb_path)):
        for file in files:
            pdb_name_set.add(file.split('.')[0])

    print('caption_name_set:', len(caption_name_set))
    print('pdb_name_set:', len(pdb_name_set))
    intersection = caption_name_set.intersection(pdb_name_set)
    difference = pdb_name_set.difference(caption_name_set)

    print('intersection:', len(intersection))
    print('have pdb but no caption:', len(difference))


if __name__ == '__main__':
    with open('./../caption-pdbs/abstract.json', 'r') as f:
        ann_json = json.load(f)
    ann_dict = dict()
    for ann in ann_json:
        ann_dict[ann['pdb_id']] = ann['caption']

    for pbd_path in tqdm(os.listdir('./../processed-pdb-dicts')):
        pdb_dict = torch.load(os.path.join('./../processed-pdb-dicts', pbd_path))
        pdb_dict['caption'] = ann_dict[pbd_path.split('.')[0]]
        torch.save(pdb_dict, os.path.join('./../processed-pdb-dicts', pbd_path))
