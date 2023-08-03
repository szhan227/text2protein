import math
import numpy as np
from pathlib import Path
import pickle as pkl
from pyrosetta import *
import rosetta_min.run as rosetta
import argparse
from tqdm import tqdm
import time
import yaml
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--index', type=int, default=1) # 1-indexing
    parser.add_argument('--pdb', type=str, default=None)
    parser.add_argument('--mask_info', type=str, default="1:5,10:15")
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--dist_std', type=float, default=2)
    parser.add_argument('--angle_std', type=float, default=20)
    parser.add_argument('--fastdesign', type=bool, default=False)
    parser.add_argument('--fastrelax', type=bool, default=False)
    args = parser.parse_args()

    print('start to run rosetta sampling...')

    ### HARD-CODED FOR PROPER NAMING ### .parent.stem
    # outPath = Path("sampling", "rosetta", args.tag,
    #                f"{Path(args.data).parent.stem}_index_{args.index}")
    # print(Path(args.data))
    # print(Path(args.data).parent)
    # print(Path(args.data).parent.parent.stem)
    # outPath = Path("sampling", "rosetta", args.tag,
    #                  f"{Path(args.data).parent.parent.stem}_index_{args.index}")
    # print(outPath)
    # return

    with open(args.data, "rb") as f:
        samples = pkl.load(f)

    # sample = samples[args.index-1]
    for i, sample in enumerate(samples):
        print('rosetta sampling #: ', i)

        outPath = Path("sampling", "rosetta", args.tag,
                          f"{Path(args.data).parent.stem}_index_{i}")

        msk = np.round(sample[-1])
        L = math.sqrt(len(msk[msk == 1]))
        if not (L).is_integer():
            raise ValueError("Terminated due to improper masking channel...")
        else:
            L = int(L)

        if args.pdb is not None:
            pose = pose_from_pdb(args.pdb)
            seq = pose.sequence()
            res_mask = args.mask_info.split(",")
            for r in res_mask:
                start_idx, end_idx = r.split(",")
                seq[int(start_idx)-1:int(end_idx)-1] = "_"
        else:
            # Initialize sequence of polyalanines and gather constraints
            seq = "A" * L
            pose = None

        npz = {}
        for idx, name in enumerate(["dist", "omega", "theta", "phi"]):
            npz[name] = np.clip(sample[idx][msk == 1].reshape(L, L), -1, 1)

        # Inverse scaling
        npz["dist_abs"] = (npz["dist"] + 1) * 10
        npz["omega_abs"] = npz["omega"] * math.pi
        npz["theta_abs"] = npz["theta"] * math.pi
        npz["phi_abs"] = (npz["phi"] + 1) * math.pi / 2

        rosetta.init_pyrosetta()

        print('Start to run minimization')
        # minimization_progress = tqdm(range(args.n_iter), desc='Minimization progress')
        # for n in minimization_progress:
        for n in range(args.n_iter):
            print(f'Minimization {n+1}/{args.n_iter} ', end='')
            start = time.time()
            outPath_run = outPath.joinpath(f"round_{n + 1}")
            if outPath_run.joinpath("final_structure.pdb").is_file():
                continue

            _ = rosetta.run_minimization(
                npz,
                seq,
                pose=pose,
                scriptdir=Path("rosetta_min"),
                outPath=outPath_run,
                angle_std=args.angle_std,  # Angular harmonic std
                dist_std=args.dist_std,  # Distance harmonic std
                use_fastdesign=args.fastdesign,
                use_fastrelax=args.fastrelax,
            )
            print(f'took {time.time()-start:.2f}s')

        # Create symlink
        if args.fastdesign:
            score_fn = create_score_function("ref2015").score
            filename = "final_structure.pdb" if args.fastrelax else "structure_after_design.pdb"
        else:
            score_fn = ScoreFunction()
            score_fn.add_weights_from_file(str(Path("rosetta_min").joinpath('data/scorefxn_cart.wts')))
            filename = "structure_before_design.pdb"

        e_min = 9999
        best_run = 0
        lowest_score_progress = tqdm(range(args.n_iter), desc='Lowest score progress')
        # for i in range(args.n_iter):

        scores = {}
        for i in lowest_score_progress:
            pose = pose_from_pdb(str(outPath.joinpath(f"round_{i + 1}", filename)))
            e = score_fn(pose)
            scores[f"round_{i + 1}"] = e
            yaml.dump({'round': i + 1, 'score': e}, open(outPath.joinpath(f"round_{i + 1}", "score.txt"), "w"))
            if e < e_min:
                best_run = i
                e_min = e
        scores["best_run"] = best_run
        yaml.dump(scores, open(outPath.joinpath("score.txt"), "w"))
        outPath.joinpath(f"best_run").symlink_to(outPath.joinpath(f"round_{best_run + 1}").resolve(),
                                                 target_is_directory=True)

        with open(outPath.joinpath(f"sample_{i}.pkl"), "wb") as f:
            pkl.dump(sample, f)

if __name__ == "__main__":
    main()
