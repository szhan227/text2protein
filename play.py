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
    import pickle as pkl
    # pdbs = ['abc', 'def', 'ggg']
    # with open('./tts.txt', 'w') as f:
    #     yaml.dump(pdbs, f)

    # path = Path('./training/test_config/2023_08_15__04_04_10/checkpoints/best.pth')
    # print(path.parent.parent.joinpath('test_ids.txt'))
    path = './processed-pdb-dicts/1sfp.pt'
    d = torch.load(path)
    print(d.keys())
    for k, v in d.items():
        if hasattr(v, 'shape'):
            print(k, v.shape)
        # elif hasattr(v, '__len__'):
        #     print(k, len(v))
        else:
            print(k, v)

    print(d['aa_str'])

    # models = []
    # pdb_paths = os.listdir('./pdbs')
    #
    # p2 = './pdbs/5e7x.pdb'
    # p1 = './pdbs/3mk9.pdb'
    # parser = PDBParser(QUIET=True)
    # s1 = parser.get_structure('pdb_structure', p1)
    # s2 = parser.get_structure('pdb_structure', p2)
    # calculate_tm_score(s1, s2)
    # for path in pdb_paths:
    #     pdb_file = './pdbs/' + path
    #     parser = PDBParser()
    #     structure = parser.get_structure('pdb_structure', pdb_file)
    #     models.append(structure)
    #
    # sctm_score = calculate_sctm_score(models)
    # print('sctm_score: ', sctm_score)
    # import yaml
    # yaml.dump({'a': 1, 'b': 2}, open('./test.txt', 'w'))
    # pdb_file = './pdbs/1sfp.pdb'
    # parser = PDBParser()
    # structure = parser.get_structure('pdb_structure', pdb_file)
    #
    # view = nv.show_biopython(structure)
    # view.clear_representations()  # Clear existing representations
    # view.add_cartoon()  # Add cartoon representation
    # view.center()  # Center the view
    # view.update_representation()  # Update the representation
    # view.show()  # Display the viewer

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    #
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Protein Structure')
    #
    # plt.show()
    # llm_name = 'NousResearch/GPT4-x-Vicuna-13b-4bit'
    # tokenizer = LlamaTokenizer.from_pretrained(llm_name, use_fast=False)
    # print('Loaded tokenizer')
    # llm = LlamaForCausalLM.from_pretrained(llm_name)
    # print('Loaded llm to cpu')
    # llm = llm.to(device)

    # raw_captions = [
    #     # 3mk9
    #     'RTA1-33/44-198 is a catalytically inactive, single-domain derivative of the ricin toxin A-chain (RTA) engineered to serve as a stable protein scaffold for presentation of native immunogenic epitopes (Olson et al., Protein Eng Des Sel 2004;17:391-397). To improve the stability and solubility of RTA1-33/44-198 further, we have undertaken the design challenge of introducing a disulfide (SS) bond. Nine pairs of residues were selected for placement of the SS-bond based on molecular dynamics simulation studies of the modeled single-domain chain. Disulfide formation at either of two positions (R48C/T77C or V49C/E99C) involving a specific surface loop (44-55) increased the protein melting temperature by ~5°C compared with RTA1-33/44-198 and by ~13°C compared with RTA. Prolonged stability studies of the R48C/T77C variant (> 60 days at 37°C, pH 7.4) confirmed a > 40% reduction in self-aggregation compared with RTA1-33/44-198 lacking the SS-bond. The R48C/T77C variant retained affinity for anti-RTA antibodies capable of neutralizing ricin toxin, including a monoclonal that recognizes a human B-cell epitope. Introduction of either R48C/T77C or V49C/E99C promoted crystallization of RTA1-33/44-198, and the X-ray structures of the variants were solved to 2.3 A or 2.1 A resolution, respectively. The structures confirm formation of an intramolecular SS-bond, and reveal a single-domain fold that is significantly reduced in volume compared with RTA. Loop 44 to 55 is partly disordered as predicted by simulations, and is positioned to form self-self interactions between symmetry-related molecules. We discuss the importance of RTA loop 34 to 55 as a nucleus for unfolding and aggregation, and draw conclusions for ongoing structure-based minimalist design of RTA-based immunogens.',
    #     # 5e7x
    #     'Talaromyces marneffei infection causes talaromycosis (previously known as penicilliosis), a very important opportunistic systematic mycosis in immunocompromised patients. Different virulence mechanisms in T. marneffei have been proposed and investigated. In the sera of patients with talaromycosis, Mp1 protein (Mp1p), a secretory galactomannoprotein antigen with two tandem ligand-binding domains (Mp1p-LBD1 and Mp1p-LBD2), was found to be abundant. Mp1p-LBD2 was reported to possess a hydrophobic cavity to bind copurified palmitic acid (PLM). It was hypothesized that capturing of lipids from human hosts by expressing a large quantity of Mp1p is a virulence mechanism of T. marneffei It was shown that expression of Mp1p enhanced the intracellular survival of T. marneffei by suppressing proinflammatory responses. Mechanistic study of Mp1p-LBD2 suggested that arachidonic acid (AA), a precursor of paracrine signaling molecules for regulation of inflammatory responses, is the major physiological target of Mp1p-LBD2. In this study, we use crystallographic and biochemical techniques to further demonstrate that Mp1p-LBD1, the previously unsolved first lipid binding domain of Mp1p, is also a strong AA-binding domain in Mp1p. These studies on Mp1p-LBD1 support the idea that the highly expressed Mp1p is an effective AA-capturing protein. Each Mp1p can bind up to 4 AA molecules. The crystal structure of Mp1p-LBD1-LBD2 has also been solved, showing that both LBDs are likely to function independently with a flexible linker between them. T. marneffei and potentially other pathogens highly expressing and secreting proteins similar to Mp1p can severely disturb host signaling cascades during proinflammatory responses by reducing the availabilities of important paracrine signaling molecules.'
    # ]
    # tokens = tokenizer(raw_captions, return_tensors="pt", add_special_tokens=False, max_length=512,
    #                    padding=True, truncation=True)
    # tokens = tokens.input_ids.to(device)
    # if hasattr(tokens, 'shape'):
    #     print('tokens shape', tokens.shape)
    # context = llm.model.embed_tokens(tokens).to(device)
    # if hasattr(context, 'shape'):
    #     print('context shape', context.shape)
    # batch = torch.load('./processed-pdb-dicts/3mk9.pt')
    # coords = batch['coords_6d']
    # print(coords.shape)
    # mask = coords[-1]
    # print(mask)
    # L = math.sqrt(len(mask[mask == 1]))
    # L = int(L)
    # print(L)
    #
    # npz = {}
    # for idx, name in enumerate(["dist", "omega", "theta", "phi"]):
    #     npz[name] = np.clip(coords[idx][mask == 1].reshape(L, L), -1, 1)
    #
    # # Inverse scaling
    # npz["dist_abs"] = (npz["dist"] + 1) * 10
    # npz["omega_abs"] = npz["omega"] * math.pi
    # npz["theta_abs"] = npz["theta"] * math.pi
    # npz["phi_abs"] = (npz["phi"] + 1) * math.pi / 2
    #
    # print(npz)
    # local_test = False
    #
    # if local_test:
    #     pdb_path = './pdbs'
    #     caption_path = './ann.json'
    # else:
    #     pdb_path = './../raw-pdbs'
    #     caption_path = './../caption-pdbs/abstract.json'
    #
    # train_ds = ProteinDataset(pdb_path, caption_path, max_res_num=256, ss_constraints=False)
    # print('train_ds:', len(train_ds))
    # train_sampler = torch.utils.data.RandomSampler(
    #     train_ds,
    #     replacement=True,
    #     num_samples=2000 * 2
    # )
    # train_dl = torch.utils.data.DataLoader(
    #     train_ds,
    #     sampler=train_sampler,
    #     batch_size=1,
    #     collate_fn=PaddingCollate(256)
    # )

    # with open('./configs/test_config_large.yml', 'r') as f:
    #     config = EasyDict(yaml.safe_load(f))
    # # # print(config)
    # #
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # # # if context shape is [B, N], then add one dimension in the mid to get [B, 1, N]
    #
    # model = UNetModel(config).to(device)
    # # print(model)
    #
    # batch = next(iter(train_dl))
    # coords_6d = batch['coords_6d'].to(device)
    # #
    # coords_6d = torch.randn(1, 5, 256, 256).to(device)
    # print('coords_6d', coords_6d.shape)
    # print('id:', batch['id'])
    # print('caption:', batch['caption'])
    #
    # # d = {
    # #     'pdb_id': '2ki6',
    # #     'caption_emb': torch.randn(1, 1, 128),
    # # }
    # #
    # # torch.save(d, './db.pt')
    #
    #
    #
    #
    #
    # timesteps = torch.randint(0, 1000, (1, )).to(device)
    # text_emb = torch.zeros(1, 77, 128).to(device)
    # print('text emb shape:', text_emb.shape)
    # output = model(coords_6d, timesteps, text_emb)
    # print(output.shape)
    #
    # sampler = DiffusionSampler(model).to(device)
    # output = sampler.ddim_sample(shape=(1, 5, 64, 64), context=text_emb)
    #
    #
    # print("after sampling:", output.shape)


    # text_emb = torch.randn(2, 256, 1024).to(device)
    # h = torch.randn(2, 512, 16, 16).to(device)
    # ch = 512
    # num_head_channels = -1
    # nums_head = 8
    # dim_head = ch // nums_head
    # trans = SpatialTransformer(in_channels=ch, n_heads=nums_head, d_head=dim_head, context_dim=1024).to(device)
    # h = trans(x=h, context=text_emb)
    # print(h.shape)






