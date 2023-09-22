import torch
from dataset import ProteinDataset, PaddingCollate
from score_sde_pytorch.models.ncsnpp import NCSNpp, UNetModel
from easydict import EasyDict
import yaml
from model.diffusion_sampler import DiffusionSampler
from model.attention import SpatialTransformer
import json
from pathlib import Path
import os
import sys
import math
import numpy as np
from transformers import LlamaTokenizer
from model.modeling_llama import LlamaForCausalLM
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Polypeptide import PPBuilder
import matplotlib.pyplot as plt
import nglview as nv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import py3Dmol
from itertools import combinations
from score_sde_pytorch.utils import save_checkpoint, restore_checkpoint, get_model, recursive_to
from torchsummary import summary
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_residue_coordinates(residue):
    return [atom.get_coord() for atom in residue.get_atoms()]


def calculate_tm_score(structure1, structure2):
    # superimposer = Superimposer()
    # atoms1 = list(structure1.get_atoms())
    # atoms2 = list(structure2.get_atoms())
    # print('atoms1: ', atoms1)
    # print('atoms2: ', atoms2)
    # superimposer.set_atoms(atoms1, atoms2)
    # return superimposer.rms
    # residues1 = [residue.get_id() for chain in structure1 for residue in chain]
    # residues2 = [residue.get_id() for chain in structure2 for residue in chain]
    #
    residues1 = [residue for chain in structure1.get_chains() for residue in chain]
    print('residues1: ', residues1)
    print('len res1: ', len(residues1))
    # print('residues2: ', residues2)


def calculate_sctm_score(models):
    total_tm_score = 0.0
    num_pairs = 0

    for i, j in combinations(range(len(models)), 2):
        print('i: ', i, 'j: ', j)
        tm_score = calculate_tm_score(models[i], models[j])
        print('tm_score: ', tm_score)
        total_tm_score += tm_score
        num_pairs += 1

    sctm_score = total_tm_score / num_pairs
    return sctm_score

if __name__ == '__main__':

    path = Path("D:\.kube").joinpath("rosetta_designed")
    sampled_names = os.listdir(path)
    print(sampled_names)
    reus = []
    for pdb_name in sampled_names:
        score_path = path.joinpath(pdb_name).joinpath("score.txt")
        if not score_path.exists():
            print('score file not exist', score_path)
            continue
        data = yaml.safe_load(open(score_path, 'r'))
        mid_name = pdb_name[1:3]
        print(f'kubectl cp siyang-pod:/siyang-storage/raw-pdbs/{mid_name}/{pdb_name}.pdb ./testpdb/{pdb_name}.pdb')
        reus.append(data['avg_score_per_res'])

    # reus.sort(reverse=True)
    # reus = reus[1:]
    # print(reus)
    # print("average REU:", sum(reus)/len(reus))
    # print("min REU:", min(reus))
    # print("max REU:", max(reus))
    # # calculate standard deviation
    # print("std REU:", np.std(reus))







