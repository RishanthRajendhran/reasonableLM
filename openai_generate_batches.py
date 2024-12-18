import pickle as pkl
from tqdm import tqdm
import argparse
import json
import logging

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

from prompts.PromptGPT import PromptGPT

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
    "--target_len",
    type=int,
    help="No. of target tokens to compute perplexity over",
    default=512
)

parser.add_argument(
    "--skip_first_n_tokens",
    type=int,
    help="No. of tokens to skip reasoning over from the start of text",
    default=64
)

parser.add_argument(
    "--out_path",
    type=str,
    help="Full directory path to save dataset"
)

parser.add_argument(
    "--skip_p",
    type=float,
    help="Probability of skipping",
    default=0.5
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

parser.add_argument(
    "--batch_path",
    type=str,
    help="Path to directory to save batches",
)

parser.add_argument(
    "-model",
    choices=[
        "gpt-4o", 
        "gpt-4o-mini", 
    ],
    help="Name of OpenAI model to use",
    default="gpt-4o-mini"
)

parser.add_argument(
    "-temperature",
    type=float,
    help="Temperature for generations",
    default=0,
)

parser.add_argument(
    "-max_tokens",
    type=int,
    help="Max no. of tokens to generate",
    default=2048,
)

parser.add_argument(
    "-top_p",
    type=float,
    help="top p for nucleus sampling",
    default=1,
)

args = parser.parse_args()
assert args.config.endswith(".json")
with open(args.config, "r") as f:
    parser.set_defaults(**json.load(f))
#---------------------------------------------------------- 
def main():
    args = parser.parse_args()

    assert (args.data_path!=None and args.out_path != None)
    check_if_exists(args.out_path, isDir=True, createIfNotExists=True)
    assert args.granularity_level!=None 
    assert args.batch_path != None
    check_if_exists(args.batch_path, isDir=True, createIfNotExists=True)

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
        cache_dir=args.cache_dir, 
        shuffle=args.shuffle,
        load_from_disk=args.load_from_disk,
        save_to_disk=not(args.load_from_disk),
        disk_path=args.out_path
    )

    data_loader = create_data_loader(
        ds, 
        args.batch_size,
        args.shuffle
    )
    
    promptGPT = PromptGPT(
        granularity_level=args.granularity_level
    )

    assert args.batch_size==1, "Only batch_size=1 supported."
    requests = []
    for id, full_text, chunk_text, start, end in tqdm(data_loader, desc="Data Loader"):
        if len(chunk_text[0]) == 0:
            continue
        for chunk_ind in range(len(chunk_text[0])):
            prompt = promptGPT.get_user_message(
                input_text=full_text[0][:start[0][chunk_ind]],
                output_text=full_text[0][start[0][chunk_ind]:end[0][chunk_ind]],
            )
            request = {
                "custom_id": "{}_{}_{}".format(id[0], start[0][chunk_ind], end[0][chunk_ind]), 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": {
                    "model": args.model, 
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user", 
                            "content": prompt.strip()
                        }
                    ],
                "temperature":args.temperature,
                "max_tokens":args.max_tokens,
                "top_p":args.top_p,
                "frequency_penalty":0,
                "presence_penalty":0,
                "response_format": {
                        "type": "text"
                    }
                }
            }
            requests.append(request)

    with open(args.batch_path + "/" + "openai_batch.jsonl", "w") as f:
        for req in requests: 
            f.write(json.dumps(req))
            f.write("\n")
#----------------------------------------------------------
if __name__ == "__main__":
    main()
