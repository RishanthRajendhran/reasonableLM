import argparse
import logging
import os
import torch
from tqdm import tqdm

import json
import numpy as np
import copy
import random

from utils.Chunker import Chunker
from utils.data_loader import create_dataset, create_data_loader
from utils.misc import check_if_exists, set_seed

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--config", 
    type=str,
    help="Path to json config file"
)

parser.add_argument(
    "--cache_dir",
    type=str,
    help="Path to HF cache",
)

parser.add_argument(
    "--dataset",
    type=str,
    help="Name of dataset",
    choices=["redpajama"],
    default="redpajama"
)

parser.add_argument(
    "--data_path",
    type=str,
    help="Path (local or HF) to directory containing HF dataset",
    default=None
)

parser.add_argument(
    "--shuffle",
    help="Shuffle dataset",
    default="False"
)

parser.add_argument(
    "--streaming",
    help="Stream dataset",
    default="False"
)

parser.add_argument(
    "--load_from_disk",
    help="Load dataset from disk",
    default="False"
)

parser.add_argument(
    "--skip_first_n_tokens",
    type=int,
    help="No. of tokens to skip reasoning over from the start of text",
    default=64
)

parser.add_argument(
    "--split",
    type=str,
    help="Split of HF dataset to use",
    default="train",
)

parser.add_argument(
    "--num_examples",
    type=int,
    help="No. of examples to take",
    default=100,
)

parser.add_argument(
    "--out_path",
    type=str,
    help="Full directory path to save dataset"
)

parser.add_argument(
    "--granularity_level",
    type=str,
    help="Level of Granularity of predictions",
    choices=["word", "phrase", "sentence", "paragraph"],
    default=None
)

parser.add_argument(
    "--granularity_len",
    type=int,
    help="Max length of granular segments",
    default=-1,
)

parser.add_argument(
    "--seed",
    type=int,
    help="Seed for torch/random/numpy",
    default=324534
)

parser.add_argument(
    "--batch_size",
    type=int,
    help="Batch size for dataloader",
    default=1
)

args = parser.parse_args()
assert args.config.endswith(".json")
with open(args.config, "r") as f:
    parser.set_defaults(**json.load(f))
#---------------------------------------------------------- 
def main():
    args = parser.parse_args()

    assert args.data_path!=None
    assert args.out_path != None
    check_if_exists(args.out_path, isDir=True, createIfNotExists=True)
    assert args.granularity_level!=None 

    if args.shuffle == "True":
        args.shuffle = True
    else: 
        args.shuffle = False

    if args.streaming == "True":
        args.streaming = True
    else: 
        args.streaming = False

    if args.load_from_disk == "True":
        args.load_from_disk = True
    else: 
        args.load_from_disk = False

    set_seed(args.seed)

    logging.basicConfig(filemode='w', level=logging.INFO)

    logging.info("Args:")
    logging.info('\n\t'.join(f'{k}={v}' for k, v in vars(args).items()))

    chunker = Chunker()
    ds = create_dataset(
        dataset=args.dataset, 
        data_path=args.data_path, 
        chunker=chunker, 
        skip_first_n_tokens=args.skip_first_n_tokens,
        granularity_level=args.granularity_level, 
        granularity_len=args.granularity_len,
        num_examples=args.num_examples,
        split=args.split,
        cache_dir=args.cache_dir, 
        streaming=args.streaming,
        shuffle=args.shuffle,
        load_from_disk=args.load_from_disk,
        save_to_disk=not(args.load_from_disk),
        disk_path=args.out_path
    )

    #Print some samples randomly
    data_loader = create_data_loader(
        ds, 
        args.batch_size,
        args.shuffle
    )

    for id, full_text, chunk_text, start, end in data_loader:
        if len(start[0])==0:
            continue
        if random.random() < 0.8:
            continue
        print("### ID\n{}".format(id[0]))
        print("### Text\n{}\n".format(full_text[0][:start[0][0]]))
        print("### Next {}\n{}\n".format(args.granularity_level, full_text[0][start[0][0]:end[0][0]]))
        print("*"*30)
#----------------------------------------------------------
if __name__ == "__main__":
    main()
