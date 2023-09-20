# Text2Protein
A text-to-protein backbone diffusion model.
## Run the code
1. Build Conda environment
   Run command:
```
conda env create -n text2protein --file env.yaml
```

2. Prepare training data

Download Protein PDBs
```
rsync -rlpt -v -z --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/pdb/ ./pdb #change ./pdb_to_your_path
```
Download Protein Abstraction at
```
https://drive.google.com/drive/u/0/folders/1yb2iP9sMyIpYsMcFvFKkjYpKmjEcPSzL
```

3. Prepare training data
Run prepare_dataset.py to generate training data from raw PDBs and abstractions. This is helpful for ruling out the proteins that are not suitable for training. Also setup your configuration. By default, use "test_config.yml"

4. Train the model:
Run train.py to train the model. In training, there is by default 95:5 train test slip, and the name of traing and test pdb will be stored in text2protein/training/{your_config_name}/{your_timestamp}/train_ids.txt and text2protein/training/{your_config_name}/{your_timestamp}/test_ids.txt

5. Sample 6D protein structures:
Run sampling_6d.py. Customize your test pdb set, taking description from the preprocessed pdbs file(by default, use the test set in the defined in train set split in training).
6. Design with PyRosetta:

Run sampling_rosetta.py. Be careful with the path of your input samples.
You will find the sampled output at text2protein/sampling/rosetta/{your_config_name}