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
from utils.all_utils import parse_response

from prompts.PromptLlama3 import PromptLlama3

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
    "--output_file",
    type=str,
    help="Full file path for output JSON file (including file name)"
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

    assert args.output_file != None
    assert args.output_file.endswith(".json")
    check_if_exists("/".join(args.output_file.split("/")[:-1]), isDir=True, createIfNotExists=True)

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

    responses = {}
    with open(args.batch_path + "/" + "openai_batch_responses.jsonl", "r") as f:
        for resp in f.readlines():
            cur_response = json.loads(resp)
            if cur_response["response"]["status_code"] != 200:
                continue
            id, start, end = cur_response["custom_id"].split("_")
            if id not in responses.keys():
                responses[id] = {}
            responses[id][start+"_"+end] = cur_response["response"]["body"]["choices"][0]["message"]["content"].strip()

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

    promptLlama3 = PromptLlama3()

    out_data = []
    assert args.batch_size==1, "Only batch_size=1 supported."
    for id, full_text, chunk_text, start, end in tqdm(data_loader, desc="Data Loader"):
        if len(chunk_text[0])==0:
            continue
        if str(id[0]) not in responses.keys():
            raise RuntimeError("Could not find id: {}".format(id[0]))
        for chunk_idx in range(len(chunk_text)):
            if str(start[0][chunk_idx])+"_"+str(end[0][chunk_idx]) not in responses[str(id[0])].keys():
                raise RuntimeError("Could not find id: {}_{}".format(start[0][chunk_idx], end[0][chunk_idx]))
            explanation = parse_response(responses[str(id[0])][str(start[0][chunk_idx])+"_"+str(end[0][chunk_idx])])
            prompt_direct = promptLlama3.get_prompt(
                instance={
                    "text":full_text[0][:start[0][chunk_idx]],
                    "next_token":full_text[0][start[0][chunk_idx]:end[0][chunk_idx]],
                    "explanation":explanation
                }, 
                is_instruct=True, 
                granularity_level=args.granularity_level, 
                prompt_type="direct", 
                return_messages=False, 
                omit_last_body=False, 
                rationalize=False
            )
            
            prompt_answer = promptLlama3.get_prompt(
                instance={
                    "text":full_text[0][:start[0][chunk_idx]],
                    "next_token":full_text[0][start[0][chunk_idx]:end[0][chunk_idx]],
                    "explanation":explanation
                }, 
                is_instruct=True, 
                granularity_level=args.granularity_level,  
                prompt_type="explain", 
                return_messages=False, 
                omit_last_body=False, 
                rationalize=False
            )
            
            out_data.append({
                "prompt_direct": prompt_direct,
                "prompt_answer": prompt_answer,
                "outcome": 1,
                "accuracy_direct": 1,
                "accuracy_explain": 1
            })
            if random.random() > 0.8:
                logging.info("Prompt (Direct):\n{}\n".format(prompt_direct))
                logging.info("*"*20)
                logging.info("Prompt (Explain):\n{}\n".format(prompt_answer))
                logging.info("*"*40)
    #Legacy: Transform to expected format by other files
    out_data = {
        "instances": out_data
    }
    with open(args.output_file, "w") as f: 
        json.dump(out_data, f)
#----------------------------------------------------------
if __name__ == "__main__":
    main()
