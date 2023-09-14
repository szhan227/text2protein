from tmtools.io import get_structure
from tmtools import tm_align
from Bio.PDB.Polypeptide import protein_letters_3to1
import numpy as np
import os
import json
from pathlib import Path
import yaml
from tqdm import tqdm


def get_residue_data(chain):
    """Extract residue coordinates and sequence from PDB chain.

    Uses the coordinates of the Cα atom as the center of the residue.

    """
    coords = []
    seq = []
    for residue in chain.get_residues():
        if "CA" in residue.child_dict:
            coords.append(residue.child_dict["CA"].coord)
            resname = residue.resname

            # set unknown residues to X
            if resname not in protein_letters_3to1:
                resname = "X"
            else:
                resname = protein_letters_3to1[resname]
            # seq.append(protein_letters_3to1[residue.resname])
            seq.append(resname)

    return np.vstack(coords), "".join(seq)


def tm_score(target_path, reference_path):
    s1 = get_structure(target_path)
    s2 = get_structure(reference_path)

    chain1 = next(s1.get_chains())
    coords1, seq1 = get_residue_data(chain1)

    chain2 = next(s2.get_chains())
    coords2, seq2 = get_residue_data(chain2)

    res = tm_align(coords1, coords2, seq1, seq2)

    # normalize by the length of the reference pdb structure
    return res.tm_norm_chain2


def max_min_avg_tm_score(target_list, reference_list):
    scores = []
    for target_path in target_list:
        for reference_path in reference_list:
            scores.append(tm_score(target_path, reference_path))

    tm_max = max(scores)
    tm_min = min(scores)
    tm_avg = sum(scores) / len(scores)
    return tm_max, tm_min, tm_avg


if __name__ == '__main__':

    training_dir = Path('./../training/test_config/2023_08_15__04_04_10')
    rosetta_sampling_dir = Path('./../sampling/rosetta/test_config')
    train_ids_path = training_dir.joinpath('train_ids.txt')
    test_ids_path = training_dir.joinpath('test_ids.txt')

    raw_pdb_dir = Path('./../../raw-pdbs')

    train_pdb_paths = []
    with open(train_ids_path, 'r') as f_train:
        # train_ids = yaml.safe_load(f_train)
        for train_id in tqdm(f_train, desc='Loading train pdb paths'):
            train_id = train_id.strip()
            mid_name = train_id[3:5]
            train_pdb_path = raw_pdb_dir.joinpath(mid_name, f'{train_id[2:]}.pdb')
            train_pdb_paths.append(train_pdb_path)

    # for train_id in tqdm(train_ids, desc='Loading train pdb paths'):
    #     mid_name = train_id[1:3] # two character middle name
    #     train_pdb_path = raw_pdb_dir.joinpath(mid_name, f'{train_id}.pdb')
    #     if train_pdb_path.exists():
    #         train_pdb_paths.append(train_pdb_path)

    test_pdb_paths = []
    rosetta_sampling_paths = []
    with open(test_ids_path, 'r') as f_test:
        # test_ids = yaml.safe_load(f_test)
        for test_id in tqdm(f_test, desc='Loading test pdb paths'):
            test_id = test_id.strip()
            mid_name = test_id[3:5]
            test_pdb_path = raw_pdb_dir.joinpath(mid_name, f'{test_id[2:]}.pdb')
            test_pdb_paths.append(test_pdb_path)

    # for test_id in tqdm(test_ids, desc='Loading test pdb paths'):
    #     mid_name = test_id[1:3] # two character middle name
    #     test_pdb_paths.append(Path(raw_pdb_dir.joinpath(mid_name, f'{test_id}.pdb')))

        # ./text2protein/sampling/rosetta/test_config/2era/round_1/final_structure.pdb
            sampling_path = rosetta_sampling_dir.joinpath(test_id[2:], 'round_1', 'final_structure.pdb')
            if sampling_path.exists():
                rosetta_sampling_paths.append(sampling_path)

    print('Start to calculate TM score...')
    scores = []
    num_sampling = len(rosetta_sampling_paths)
    num_training = len(train_pdb_paths)
    to_save = dict()
    samples = dict()
    for i, target_path in enumerate(rosetta_sampling_paths):
        sample_name = "sampled_" + target_path.parent.parent.name
        sampled_scores = []

        progress_bar = tqdm(enumerate(train_pdb_paths))
        err = 0
        for j, reference_path in progress_bar:
            # reference_path = os.path.join(raw_pdb_dir, reference_path[:-3] + '.pdb')
            try:
                score = tm_score(target_path, reference_path)
                scores.append(score)
                sampled_scores.append(score)
                progress_bar.set_description(f'Calculating TM score: {i + 1}/{num_sampling}, {j + 1}/{num_training}, [err: {err}]')
            except Exception as e:
                err += 1
                # print('catch exception in tm_score, but ignore it.')
        if len(sampled_scores) > 0:
            sample_min = min(sampled_scores)
            sample_max = max(sampled_scores)
            sample_avg = sum(sampled_scores) / len(sampled_scores)
            samples[sample_name] = dict(sample_min=sample_min,
                                        sample_max=sample_max,
                                        sample_avg=sample_avg)
    print()

    to_save['samples'] = samples
    tm_max = max(scores)
    tm_min = min(scores)
    tm_avg = sum(scores) / max(1, len(scores))
    print(tm_max, tm_min, tm_avg)
    to_save['samples'] = samples
    to_save['tm_max'] = tm_max
    to_save['tm_min'] = tm_min
    to_save['tm_avg'] = tm_avg
    to_save['reference_count'] = len(train_pdb_paths)
    to_save['target_count'] = len(rosetta_sampling_paths)
    to_save = dict(tm_max=tm_max,
                   tm_min=tm_min,
                   tm_avg=tm_avg,
                   reference_count=len(train_pdb_paths),
                   target_count=len(rosetta_sampling_paths)
                   )

    with open('tm-scores.json', 'w') as f:
        json.dump(to_save, f, indent=4)

