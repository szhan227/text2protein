from pathlib import Path
import score_sde_pytorch.losses as losses
import score_sde_pytorch.sampling as sampling
import argparse
from score_sde_pytorch.models.ema import ExponentialMovingAverage
import score_sde_pytorch.sde_lib as sde_lib
import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from score_sde_pytorch.utils import save_checkpoint, restore_checkpoint, get_model, recursive_to
from dataset import ProteinDataset, ProteinProcessedDataset, PaddingCollate
import pickle as pkl
import yaml
from easydict import EasyDict
import time
from utils import random_mask_batch, get_condition_from_batch
import shutil
from model.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from tqdm import tqdm

def main(rank):
    print('in main: show rank:', rank)
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--local_test', type=bool, default=False)
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()


    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    device = config.device

    ss_constraints = True if config.data.num_channels == 8 else False

    dataset_path = config.data.dataset_path
    caption_path = config.data.caption_path
    processed_dataset_path = config.data.processed_dataset_path

    dataset = ProteinProcessedDataset(processed_dataset_path)



    train_size = max(1, int(0.95 * len(dataset)))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                      generator=torch.Generator().manual_seed(config.seed))



    train_dl = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        batch_size=config.training.batch_size,
        collate_fn=PaddingCollate(config.data.max_res_num)
    )

    test_sampler = torch.utils.data.RandomSampler(
        test_ds,
        replacement=True,
        num_samples=config.training.n_iters * config.training.batch_size
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        sampler=test_sampler,
        batch_size=config.training.batch_size,
        collate_fn=PaddingCollate(config.data.max_res_num)
    )

    # Create directories for experimental logs
    if args.resume is not None:
        workdir = Path(args.resume)
    else:
        workdir = Path("training", Path(args.config).stem, time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))
        workdir.mkdir(exist_ok=True,parents=True)
        # Save config to workdir
        shutil.copy(args.config, workdir.joinpath("config.yml"))

    train_ids = []
    test_ids = []
    for each in tqdm(train_ds, desc='save train ids'):
        train_pdb_id = each['id']
        train_ids.append(train_pdb_id)
    for each in tqdm(test_ds, desc='save test ids'):
        test_pdb_id = each['id']
        test_ids.append(test_pdb_id)

    with open(workdir.joinpath('train_ids.txt'), 'w') as f:
        yaml.dump(train_ids, f)
    with open(workdir.joinpath('test_ids.txt'), 'w') as f:
        yaml.dump(test_ids, f)

    sample_dir = workdir.joinpath("samples")
    sample_dir.mkdir(exist_ok=True)

    tb_dir = workdir.joinpath("tensorboard")
    tb_dir.mkdir(exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = get_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    llm_name = 'lmsys/vicuna-7b-v1.3'
    tokenizer = LlamaTokenizer.from_pretrained(llm_name, use_fast=False)
    print('Loaded tokenizer')
    llm = LlamaForCausalLM.from_pretrained(llm_name)
    print('Loaded llm to cpu')


    state = dict(optimizer=optimizer, model=score_model, llm=(tokenizer, llm), ema=ema, step=0)
    # Create checkpoints directory
    checkpoint_dir = workdir.joinpath("checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = workdir.joinpath("checkpoints-meta", "checkpoint.pth")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_meta_dir.parent.mkdir(exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    if checkpoint_meta_dir.is_file():
        state = restore_checkpoint(checkpoint_meta_dir, state, device)
        initial_step = int(state['step'])
    else:
        initial_step = 0

    print(f"Starting from step {initial_step}...")

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions

    optimize_fn = losses.optimization_manager(config)
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn)

    # # # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.max_res_num, config.data.max_res_num)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)

    min_avg_train_loss = 1e10
    train_last_save_epoch = -1
    train_batch_num = len(train_dl)
    test_batch_num = len(test_dl)

    min_avg_eval_loss = 1e10
    eval_last_save_epoch = -1
    eval_batch_num = len(test_dl)

    for epoch in range(config.training.epochs):

        all_train_losses = []
        all_eval_losses = []
        train_progress_bar = tqdm(train_dl)
        eval_progress_bar = tqdm(test_dl)
        update_train_best = False
        update_eval_best = False

        # ------------------------ Training -----------------------
        for step, batch in enumerate(train_progress_bar):
            batch = recursive_to(batch, device)
            # Execute one training step
            batch = random_mask_batch(batch, config)
            loss = train_step_fn(state, batch, condition=config.model.condition)
            all_train_losses.append(loss.item())
            avg_loss = sum(all_train_losses) / len(all_train_losses)
            train_progress_bar.set_description(f"Epoch: {epoch}, Step: {step + 1}/{train_batch_num}, batch_loss: {loss.item()}, avg_loss: {avg_loss}")
            cur_step = epoch * train_batch_num + step
            if cur_step % config.training.log_freq == 0:
                writer.add_scalar("training_loss", loss, cur_step)

        save_checkpoint(checkpoint_meta_dir, state)



        # -----------------------------Evaluation---------------------------------
        # Report the loss on an evaluation dataset periodically
        for eval_step, eval_batch in enumerate(eval_progress_bar):
            eval_batch = recursive_to(eval_batch, device)
            eval_batch = random_mask_batch(eval_batch, config)
            eval_loss = eval_step_fn(state, eval_batch, condition=config.model.condition)
            all_eval_losses.append(eval_loss.item())

        # Generate and save samples
        if config.training.snapshot_sampling:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            condition = get_condition_from_batch(config, eval_batch)

            raw_captions = [
                # 3mk9
                'RTA1-33/44-198 is a catalytically inactive, single-domain derivative of the ricin toxin A-chain (RTA) engineered to serve as a stable protein scaffold for presentation of native immunogenic epitopes (Olson et al., Protein Eng Des Sel 2004;17:391-397). To improve the stability and solubility of RTA1-33/44-198 further, we have undertaken the design challenge of introducing a disulfide (SS) bond. Nine pairs of residues were selected for placement of the SS-bond based on molecular dynamics simulation studies of the modeled single-domain chain. Disulfide formation at either of two positions (R48C/T77C or V49C/E99C) involving a specific surface loop (44-55) increased the protein melting temperature by ~5°C compared with RTA1-33/44-198 and by ~13°C compared with RTA. Prolonged stability studies of the R48C/T77C variant (> 60 days at 37°C, pH 7.4) confirmed a > 40% reduction in self-aggregation compared with RTA1-33/44-198 lacking the SS-bond. The R48C/T77C variant retained affinity for anti-RTA antibodies capable of neutralizing ricin toxin, including a monoclonal that recognizes a human B-cell epitope. Introduction of either R48C/T77C or V49C/E99C promoted crystallization of RTA1-33/44-198, and the X-ray structures of the variants were solved to 2.3 A or 2.1 A resolution, respectively. The structures confirm formation of an intramolecular SS-bond, and reveal a single-domain fold that is significantly reduced in volume compared with RTA. Loop 44 to 55 is partly disordered as predicted by simulations, and is positioned to form self-self interactions between symmetry-related molecules. We discuss the importance of RTA loop 34 to 55 as a nucleus for unfolding and aggregation, and draw conclusions for ongoing structure-based minimalist design of RTA-based immunogens.',
                # 5e7x
                'Talaromyces marneffei infection causes talaromycosis (previously known as penicilliosis), a very important opportunistic systematic mycosis in immunocompromised patients. Different virulence mechanisms in T. marneffei have been proposed and investigated. In the sera of patients with talaromycosis, Mp1 protein (Mp1p), a secretory galactomannoprotein antigen with two tandem ligand-binding domains (Mp1p-LBD1 and Mp1p-LBD2), was found to be abundant. Mp1p-LBD2 was reported to possess a hydrophobic cavity to bind copurified palmitic acid (PLM). It was hypothesized that capturing of lipids from human hosts by expressing a large quantity of Mp1p is a virulence mechanism of T. marneffei It was shown that expression of Mp1p enhanced the intracellular survival of T. marneffei by suppressing proinflammatory responses. Mechanistic study of Mp1p-LBD2 suggested that arachidonic acid (AA), a precursor of paracrine signaling molecules for regulation of inflammatory responses, is the major physiological target of Mp1p-LBD2. In this study, we use crystallographic and biochemical techniques to further demonstrate that Mp1p-LBD1, the previously unsolved first lipid binding domain of Mp1p, is also a strong AA-binding domain in Mp1p. These studies on Mp1p-LBD1 support the idea that the highly expressed Mp1p is an effective AA-capturing protein. Each Mp1p can bind up to 4 AA molecules. The crystal structure of Mp1p-LBD1-LBD2 has also been solved, showing that both LBDs are likely to function independently with a flexible linker between them. T. marneffei and potentially other pathogens highly expressing and secreting proteins similar to Mp1p can severely disturb host signaling cascades during proinflammatory responses by reducing the availabilities of important paracrine signaling molecules.'
            ]
            tokens = tokenizer(raw_captions, return_tensors="pt", add_special_tokens=False, max_length=512, padding=True, truncation=True)
            tokens = tokens.input_ids
            context = llm.model.embed_tokens(tokens).to(device)
            # context = torch.randn(config.training.batch_size, 1, 128)

            sample, n = sampling_fn(score_model, condition=condition, context=context)
            ema.restore(score_model.parameters())
            this_sample_dir = sample_dir.joinpath(f"epoch_{epoch}")
            this_sample_dir.mkdir(exist_ok=True)

            with open(str(this_sample_dir.joinpath("sample.pkl")), "wb") as fout:
                pkl.dump(sample.cpu(), fout)

            # save_grid(sample.cpu().numpy(), this_sample_dir.joinpath("sample.png"))

        if len(all_train_losses) > 0:
            avg_train_loss = sum(all_train_losses) / len(all_train_losses)

            # write average training loss to tensorboard after each epoch
            writer.add_scalar("avg_training_loss", avg_train_loss, epoch)

            if avg_train_loss < min_avg_train_loss:
                min_avg_train_loss = avg_train_loss
                print(f'Train: Update best training model at epoch {epoch}, eval_avg_loss:', avg_train_loss)
                save_checkpoint(checkpoint_dir.joinpath(f'best_train.pth'), state)
                train_last_save_epoch = epoch
                update_train_best = True
            else:
                print(f'Train: Not update best model at epoch {epoch}, cur_avg_loss:', avg_train_loss,
                      'min_avg_loss:', min_avg_train_loss,
                      'seen at epoch:', train_last_save_epoch)

        if len(all_eval_losses) > 0:
            avg_eval_loss = sum(all_eval_losses) / len(all_eval_losses)

            writer.add_scalar("avg_eval_loss", avg_eval_loss, epoch)

            if avg_eval_loss < min_avg_eval_loss:
                min_avg_eval_loss = avg_eval_loss
                print(f'Eval: Update best eval model at epoch {epoch}, eval_avg_loss:', avg_eval_loss)
                save_checkpoint(checkpoint_dir.joinpath(f'best_eval.pth'), state)
                eval_last_save_epoch = epoch
                update_eval_best = True
            else:
                print(f'Eval: Not update best model at epoch {epoch}, cur_avg_loss:', avg_eval_loss,
                        'min_avg_loss:', min_avg_eval_loss,
                        'seen at epoch:', eval_last_save_epoch)



if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    n_gpus = 1
    print('n_gpus', n_gpus)
    if n_gpus <= 1:
        main(0)
    else:
        torch.multiprocessing.spawn(main, nprocs=n_gpus)
