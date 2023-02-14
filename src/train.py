"""
Run GM-VAE Training
"""
import os
import json
import torch
from datetime import datetime

import sys  # Todo: fix this
sys.path.append("..")

from common.utils import collate_skip_none
from common.constants import GM_VAE_DIRS

from mt_gm_vae.trainer import GMVAE_Trainer
from mt_gm_vae.models import MT_GMVAE, MTConfig
from gm_vae.data_loader import MusicDataset


# Example Configuration
train_config = {"lr": 0.001,
                "batch_size": 64,
                "n_epochs": 100,
                "print_step": 100,
                "beta": 0.1}

# Use timestamp as file name prefix
timestamp = datetime.now().strftime("%y%m%dT%H")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", dest="data_dir", type=str,
                        default=os.path.join(GM_VAE_DIRS["data"], "data"),
                        help="Path to preprocessed train/validation data.")
    parser.add_argument("--train_conf", dest="train_conf", type=str,
                        default="./train_config.json",
                        help="Json file for training configuration. See `train_config.json` for details.")
    parser.add_argument("--model", dest="model", type=str,
                        default=os.path.join(GM_VAE_DIRS["models"], f"mt_{timestamp}.pt"), help="Directory to save/saved models.")
    parser.add_argument("--supervised", dest="supervised", default=True, action="store_false",
                        help="Train/test supervised model.")
    parser.add_argument("-y", dest="y_label", type=str,
                        default="arousal", help="`arousal/valence` if is supervised.")

    # Optional
    parser.add_argument("--num_workers", dest="num_workers", default=0, type=int,
                        help='Number of workers for dataloading (default: 0, loading on main process)')

    args = parser.parse_args()

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {str(device)}\n')

    # Load Configuration
    with open(args.train_conf, "r") as f:
        train_config = json.load(f)

    # Data Directories
    train_data_dir = os.path.join(args.data_dir, "train")
    valid_data_dir = os.path.join(args.data_dir, "valid")

    # Check pre-processed data existance
    if os.path.exists(train_data_dir) and os.path.exists(valid_data_dir):
        print("Loading Data from")
        print(f"Train: {train_data_dir}")
        print(f"Val: {valid_data_dir}\n")
    else:
        print(f"Data not found: {train_data_dir}, {valid_data_dir}")
        sys.exit(1)

    # Data Loader
    train_dl = torch.utils.data.DataLoader(
        MusicDataset(train_data_dir,
                     supervised=args.supervised,
                     y_label=args.y_label,
                     input_feat="spectrogram"),
        collate_fn=collate_skip_none,
        batch_size=train_config["batch_size"],
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_dl = torch.utils.data.DataLoader(
        MusicDataset(valid_data_dir,
                     supervised=args.supervised,
                     y_label=args.y_label,
                    input_feat="spectrogram"),
        collate_fn=collate_skip_none,
        batch_size=train_config["batch_size"],
        shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # Build Model
    model = MT_GMVAE(MTConfig)

    # Start Training
    gm_vae_trainer = GMVAE_Trainer(model, device, supervised=args.supervised)
    gm_vae_trainer.train(train_dl, valid_dl, args.model, postfix=timestamp, **train_config)
