import torch
from pathlib import Path
from score_sde_pytorch.utils import get_model, restore_checkpoint, recursive_to
from score_sde_pytorch.models.ema import ExponentialMovingAverage
import score_sde_pytorch.sde_lib as sde_lib
import score_sde_pytorch.sampling as sampling
import score_sde_pytorch.losses as losses
import pickle as pkl
import argparse
import yaml
from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict
from tqdm.auto import tqdm
from utils import get_conditions_random, get_mask_all_lengths, get_conditions_from_pdb
from model.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from dataset import ProteinProcessedDataset
import os


class DescriptionDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    # captions = [['A', 'aaa'], ['B', 'bbbbb'], ['C', 'ccccccccc'], ['D', 'dfdfdf'], ['E', 'asdasdasd'], ['F', 'fdsfsdfdsf'], ['G', 'fffffffggggg']]
    # ds = DescriptionDataset(captions)
    # dl = DataLoader(ds, batch_size=3, shuffle=False)
    # for id, text in dl:
    #     print(list(text))
    # return

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--pdb', type=str, default=None)
    parser.add_argument('--chain', type=str, default="A")
    parser.add_argument('--mask_info', type=str, default="1:5,10:15")
    parser.add_argument('--tag', type=str, default="test")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--select_length', type=bool, default=False)
    parser.add_argument('--length_index', type=int, default=1) # Index starts at 1
    args = parser.parse_args()

    assert not (args.pdb is not None and args.select_length)

    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    config.device = args.device
    workdir = Path("sampling", "coords_6d", Path(args.config).stem, Path(args.checkpoint).parent.parent.stem, args.tag)

    # Initialize model.
    device = config.device
    score_model = get_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    state = restore_checkpoint(args.checkpoint, state, args.device)
    state['ema'].store(state["model"].parameters())
    state['ema'].copy_to(state["model"].parameters())

    # Load SDE
    if config.training.sde == "vesde":
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    elif config.training.sde == "vpsde":
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3

    # Sampling function
    sampling_shape = (args.batch_size, config.data.num_channels,
                      config.data.max_res_num, config.data.max_res_num)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)

    # pdb_des_3mk9 = 'RTA1-33/44-198 is a catalytically inactive, single-domain derivative of the ricin toxin A-chain (RTA) engineered to serve as a stable protein scaffold for presentation of native immunogenic epitopes (Olson et al., Protein Eng Des Sel 2004;17:391-397). To improve the stability and solubility of RTA1-33/44-198 further, we have undertaken the design challenge of introducing a disulfide (SS) bond. Nine pairs of residues were selected for placement of the SS-bond based on molecular dynamics simulation studies of the modeled single-domain chain. Disulfide formation at either of two positions (R48C/T77C or V49C/E99C) involving a specific surface loop (44-55) increased the protein melting temperature by ~5°C compared with RTA1-33/44-198 and by ~13°C compared with RTA. Prolonged stability studies of the R48C/T77C variant (> 60 days at 37°C, pH 7.4) confirmed a > 40% reduction in self-aggregation compared with RTA1-33/44-198 lacking the SS-bond. The R48C/T77C variant retained affinity for anti-RTA antibodies capable of neutralizing ricin toxin, including a monoclonal that recognizes a human B-cell epitope. Introduction of either R48C/T77C or V49C/E99C promoted crystallization of RTA1-33/44-198, and the X-ray structures of the variants were solved to 2.3 A or 2.1 A resolution, respectively. The structures confirm formation of an intramolecular SS-bond, and reveal a single-domain fold that is significantly reduced in volume compared with RTA. Loop 44 to 55 is partly disordered as predicted by simulations, and is positioned to form self-self interactions between symmetry-related molecules. We discuss the importance of RTA loop 34 to 55 as a nucleus for unfolding and aggregation, and draw conclusions for ongoing structure-based minimalist design of RTA-based immunogens.'
    # pdb_des_5e7x = 'Talaromyces marneffei infection causes talaromycosis (previously known as penicilliosis), a very important opportunistic systematic mycosis in immunocompromised patients. Different virulence mechanisms in T. marneffei have been proposed and investigated. In the sera of patients with talaromycosis, Mp1 protein (Mp1p), a secretory galactomannoprotein antigen with two tandem ligand-binding domains (Mp1p-LBD1 and Mp1p-LBD2), was found to be abundant. Mp1p-LBD2 was reported to possess a hydrophobic cavity to bind copurified palmitic acid (PLM). It was hypothesized that capturing of lipids from human hosts by expressing a large quantity of Mp1p is a virulence mechanism of T. marneffei It was shown that expression of Mp1p enhanced the intracellular survival of T. marneffei by suppressing proinflammatory responses. Mechanistic study of Mp1p-LBD2 suggested that arachidonic acid (AA), a precursor of paracrine signaling molecules for regulation of inflammatory responses, is the major physiological target of Mp1p-LBD2. In this study, we use crystallographic and biochemical techniques to further demonstrate that Mp1p-LBD1, the previously unsolved first lipid binding domain of Mp1p, is also a strong AA-binding domain in Mp1p. These studies on Mp1p-LBD1 support the idea that the highly expressed Mp1p is an effective AA-capturing protein. Each Mp1p can bind up to 4 AA molecules. The crystal structure of Mp1p-LBD1-LBD2 has also been solved, showing that both LBDs are likely to function independently with a flexible linker between them. T. marneffei and potentially other pathogens highly expressing and secreting proteins similar to Mp1p can severely disturb host signaling cascades during proinflammatory responses by reducing the availabilities of important paracrine signaling molecules.'
    # pdb_des_6wdp = 'Interleukin-12 (IL-12) and IL-23 are heterodimeric cytokines that are produced by antigen-presenting cells to regulate the activation and differentiation of lymphocytes, and they share IL-12Rβ1 as a receptor signaling subunit. We present a crystal structure of the quaternary IL-23 (IL-23p19/p40)/IL-23R/IL-12Rβ1 complex, together with cryoelectron microscopy (cryo-EM) maps of the complete IL-12 (IL-12p35/p40)/IL-12Rβ2/IL-12Rβ1 and IL-23 receptor (IL-23R) complexes, which reveal "non-canonical" topologies where IL-12Rβ1 directly engages the common p40 subunit. We targeted the shared IL-12Rβ1/p40 interface to design a panel of IL-12 partial agonists that preserved interferon gamma (IFNγ) induction by CD8 + T cells but impaired cytokine production from natural killer (NK) cells in vitro. These cell-biased properties were recapitulated in vivo, where IL-12 partial agonists elicited anti-tumor immunity to MC-38 murine adenocarcinoma absent the NK-cell-mediated toxicity seen with wild-type IL-12. Thus, the structural mechanism of receptor sharing used by IL-12 family cytokines provides a protein interface blueprint for tuning this cytokine axis for therapeutics.'

    test_captions = []

    train_paths = []
    test_paths = []

    # ./ training / test_config / 2023_08_15__04_04_10 / checkpoints / best.pth
    chk_path = Path(args.checkpoint).parent.parent # / training / test_config / 2023_08_15__04_04_10
    with open(chk_path.joinpath('test_ids.txt', 'r')) as f:
        test_ids = yaml.safe_load(f)
    for test_id in test_ids:
        test_paths.append(os.path.join('./../processed-all-pdb-dicts', test_id + '.pt'))

    with open(chk_path.joinpath('train_ids.txt', 'r')) as f:
        train_ids = yaml.safe_load(f)
    for train_id in train_ids:
        train_paths.append(os.path.join('./../processed-all-pdb-dicts', train_id + '.pt'))

    for test_path in test_paths:
        test_dict = torch.load(test_path)
        stem = Path(test_path).stem
        test_captions.append((stem, test_dict['description']))


    description_dataset = DescriptionDataset(test_captions)
    description_loader = DataLoader(description_dataset, batch_size=args.batch_size, shuffle=True)

    llm_name = 'lmsys/vicuna-7b-v1.3'
    tokenizer = LlamaTokenizer.from_pretrained(llm_name, use_fast=False)
    print('Loaded tokenizer')
    llm = LlamaForCausalLM.from_pretrained(llm_name)
    print('Loaded llm to cpu')

    count = 0
    total = len(description_loader)
    for pdb_id, raw_caption in description_loader:
        count += 1
        if len(pdb_id) != args.batch_size:
            continue

        tokens = tokenizer(list(raw_caption), return_tensors="pt", add_special_tokens=False, max_length=512, padding=True,
                           truncation=True)
        tokens = tokens.input_ids
        context = llm.model.embed_tokens(tokens).to(device)
        # context = torch.zeros(1, 512, 4096).to(device)

        generated_samples = []
        print('start sampling')

        # for _ in tqdm(range(args.n_iter)):
        if args.select_length:
            mask = get_mask_all_lengths(config,batch_size=args.batch_size)[args.length_index-1]
            condition = {"length": mask.to(config.device)}
        elif args.pdb is not None:
            condition = get_conditions_from_pdb(args.pdb, config, args.chain, args.mask_info, batch_size=args.batch_size)
        else:
            # condition = get_conditions_random(config, batch_size=args.batch_size)
            condition = {}
        sample, n = sampling_fn(state["model"], condition=condition, context=context)
        generated_samples.append(sample.cpu())

        generated_samples = torch.cat(generated_samples, 0)
        print('show generated samples shape: ', generated_samples.shape)

        workdir.mkdir(parents=True, exist_ok=True)

        # print(f'[{count} / {total}]save samples to ', workdir.joinpath(f"sampled_{pdb_id}.pkl"))
        for i in range(args.batch_size):
            with open(workdir.joinpath(f"sampled_{pdb_id[i]}.pkl"), "wb") as f:
                pkl.dump(generated_samples[i].unsqueeze(0), f)
        print(f'[{count} / {total}] save samples.')

if __name__ == "__main__":
    main()
