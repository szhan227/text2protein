from tmtools.io import get_structure
from tmtools import tm_align
from Bio.PDB.Polypeptide import protein_letters_3to1
import numpy as np
import os
from pathlib import Path
import yaml

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
    raw_pdb_dir = Path('./../../raw-pdbs')
    # train_pdb_dir = Path('./../../processed-all-pdb-pdbs')
    train_pdb_dir = Path('./../pdbs')
    train_stems = os.listdir(train_pdb_dir)
    sampling_paths = []

    scores = []
    for target_path in sampling_paths:
        for reference_path in train_stems:
            reference_path = os.path.join(raw_pdb_dir, reference_path[:-3] + '.pdb')
            scores.append(tm_score(target_path, reference_path))

    tm_max = max(scores)
    tm_min = min(scores)
    tm_avg = sum(scores) / max(1, len(scores))
    print(tm_max, tm_min, tm_avg)
    to_save = dict(tm_max=tm_max,
                   tm_min=tm_min,
                   tm_avg=tm_avg,
                   reference_count=len(train_stems),
                   target_count=len(sampling_paths)
                   )

    with open('tm-scores.yaml', 'w') as f:
        yaml.dump(to_save, f)