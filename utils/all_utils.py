import torch
from torch.utils.data import DataLoader

from transformers import LlamaForCausalLM

# import numpy as np
import logging
import os
import psutil
import subprocess
import gc

from datasets import load_dataset, load_from_disk
import pandas as pd
import pickle as pkl
import json
import csv

from pathlib import Path
from os.path import exists
import os
import glob
#---------------------------------------------------------- 
# def get_target_neg_logprobs(model, tokenizer, input_ids, target_ids, device):
#     generated_ids = input_ids.to(device)

#     target_neg_logprobs = []
#     meta = {
#         "max_entropy": torch.log(torch.tensor(tokenizer.vocab_size)),
#         "entropy": [],
#         "generated_ids": generated_ids,
#     }

#     for idx in range(target_ids.shape[-1]):
#         outputs = model(generated_ids)
#         next_token_logits = outputs.logits[:, -1, :]
#         next_token_probs = torch.softmax(next_token_logits, dim=-1)
#         next_token_logprobs = torch.log(next_token_probs)
        
#         meta["entropy"].append(-torch.sum(next_token_probs * next_token_logprobs))

#         target_neg_logprobs.append(-1*(next_token_logprobs[:, target_ids[idx]]))

#         next_token_id = next_token_logits.argmax(dim=-1)

#         generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=1)
        
#         meta["generated_ids"] = generated_ids

#         # if next_token_id.item() == tokenizer.eos_token_id:
#         #     break
    
#     meta["entropy"] = torch.stack(meta["entropy"])
#     return torch.stack(target_neg_logprobs), meta
#---------------------------------------------------------- 
def get_entropy(logits, target_ids, tokenizer=None):
    entropy = []
    target_inds = torch.where(target_ids!= -100)
    target_rows = target_inds[0].tolist()
    target_cols = target_inds[-1].tolist()
    predicted_next_token = ""
    target_rank = []
    for row_idx, col_idx in zip(target_rows, target_cols):
        #Target tokens are one-shifted 
        next_token_logits = logits[row_idx, col_idx-1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        if tokenizer != None:
            target_rank.append((torch.argsort(next_token_probs, descending=True) == target_ids[row_idx, col_idx].item()).nonzero(as_tuple=True)[0].item())
            predicted_token_id = torch.argmax(next_token_probs, dim=-1)
            predicted_next_token += tokenizer.decode(predicted_token_id)
        next_token_logprobs = torch.log(next_token_probs)
        entropy.append(-torch.sum(next_token_probs * next_token_logprobs))
    entropy = torch.stack(entropy)
    if tokenizer != None:
        return entropy, predicted_next_token, sum(target_rank)/len(target_rank)
    else:
        return entropy
#---------------------------------------------------------- 
def parse_response(response):
    str_to_check_for = [
        "### Final prediction",
        "### Final Prediction",
        "###Final prediction",
        "###Final Prediction",
    ]
    for str_check in str_to_check_for:
        if str_check in response:
            response = response[:response.find(str_check)]
            break
    response = response.strip()

    return response
#---------------------------------------------------------- 
def mark_target_tokens(next_token_ids, target_ids, tokenizer, setting):
    #Only compute loss over the target tokens
    if setting=="base":
        for i in range(next_token_ids.shape[-1]):
            assert next_token_ids[i].item() == target_ids[0, -(next_token_ids.shape[-1]-i)].item(), "({}) Expected {}, found {}".format(setting, tokenizer.decode(next_token_ids[i].item()), tokenizer.decode(target_ids[0, -(next_token_ids.shape[-1]-i)].item()))
        target_ids[0, :-(next_token_ids.shape[-1])] = -100
    elif setting in ["direct"]:
        #Base format
        #Assumption: next_token is the last token
        # next_token_sanity = ""
        for i in range(next_token_ids.shape[-1]):
            # next_token_sanity += tokenizer.decode(next_token_ids[i].item())
            assert next_token_ids[i].item() == target_ids[0, -(next_token_ids.shape[-1]-i)].item(), "({}) Expected {}, found {}".format(setting, tokenizer.decode(next_token_ids[i].item()), tokenizer.decode(target_ids[0, -(next_token_ids.shape[-1]-i)].item()))
        # logging.info("({}) next_token_sanity: {}".format(setting, next_token_sanity))
        target_ids[0, :-(next_token_ids.shape[-1])] = -100
    elif setting in ["explain", "answer"]:
        #User-Assistant multi-turn format
        #Assumption: next_token is the second last token followed by <|eot_id|> token
        # next_token_sanity = ""
        for i in range(next_token_ids.shape[-1]):
            # next_token_sanity += tokenizer.decode(next_token_ids[i].item())
            assert next_token_ids[i].item() == target_ids[0, -(next_token_ids.shape[-1]-i+1)].item(), "({}) Expected {}, found {}".format(setting, tokenizer.decode(next_token_ids[i].item()), tokenizer.decode(target_ids[0, -(next_token_ids.shape[-1]-i+1)].item()))
        # logging.info("({}) next_token_sanity: {}".format(setting, next_token_sanity))
        target_ids[0, :-(next_token_ids.shape[-1]+1)] = -100
        #Ignore <|eot_id|> token
        assert target_ids[0, -1].item() == tokenizer.eos_token_id, "({}) Expected <|eot_id|>, found {}".format(setting, tokenizer.decode(target_ids[0, -1]))
        target_ids[0, -1] = -100
    else: 
        raise ValueError("Unrecognized setting: {}".format(setting))

    return target_ids
#---------------------------------------------------------- 
def load_model(model_path, tokenizer, model_args={}):
    if "llama" in model_path.lower():
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            dtype=model_args["dtype"] if model_args.get("dtype") else torch.bfloat16,
            device_map=model_args["device_map"] if model_args.get("device_map") else "balanced",
            cache_dir=model_args["cache_dir"] if model_args.get("cache_dir") else None,
            attn_implementation=model_args["attn_implementation"] if model_args.get("attn_implementation") else None,
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    else: 
        raise ValueError("Model not supported:".format(model_path))
    return model
#---------------------------------------------------------- 