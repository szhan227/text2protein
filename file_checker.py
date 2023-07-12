import os
import json
from pathlib import Path


def compare_pdb_file_and_caption():
    caption_path = './../caption-pdbs/abstract.json'
    pdb_path = './../raw-pdbs'

    caption_name_set = set()
    pdb_name_set = set()

    with open(caption_path, 'r') as json_file:
        # here json format: key=pdb_id, value=caption_embedding
        ann_json = json.load(json_file)
    for ann in ann_json:
        caption_name_set.add(ann['pdb_id'])

    for root, dirs, files in os.walk(pdb_path):
        for file in files:
            pdb_name_set.add(file)

    print('caption_name_set:', len(caption_name_set))
    print('pdb_name_set:', len(pdb_name_set))
    intersection = caption_name_set.intersection(pdb_name_set)
    difference = pdb_name_set.difference(caption_name_set)

    print('intersection:', len(intersection))
    print('have pdb but no caption:', len(difference))


if __name__ == '__main__':
    compare_pdb_file_and_caption()
