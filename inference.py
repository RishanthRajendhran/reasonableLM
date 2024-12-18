import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import argparse
import json
import numpy as np
import copy
import logging
import os
import time
import random

from utils.Chunker import Chunker
from utils.data_loader import create_dataset, create_data_loader
from utils.misc import check_if_exists, set_seed, clear_memory
from utils.all_utils import mark_target_tokens, get_entropy, parse_response

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

parser.add_argument(
    "--model_name",
    type=str,
    help="Path to HF Llama model",
    default=None
    # required=True
)

parser.add_argument(
    "--direct_model_name",
    type=str,
    help="Path to HF Llama model for Direct prompting",
    default=None 
)

parser.add_argument(
    "--explain_model_name",
    type=str,
    help="Path to HF Llama model for Explain prompting",
    default=None
)

parser.add_argument(
    "--max_length",
    type=int,
    help="Max length of prompt (in no. of tokens)",
    default=-1
)

parser.add_argument(
    "--model_max_length",
    type=int,
    help="Model max context length",
    default=128000
)

parser.add_argument(
    "--target_len",
    type=int,
    help="No. of target tokens to compute perplexity over",
    default=512
)

parser.add_argument(
    "--max_new_tokens",
    type=int,
    help="Max length of generated explanation",
    default=256
)

parser.add_argument(
    "--num_beams",
    type=int,
    help="No. of beams to maintain during beam search while generating explanation",
    default=1
)

parser.add_argument(
    "--do_sample",
    type=str,
    help="Set to false for greedy decoding",
    choices=["True", "False"],
    default="False"
)

parser.add_argument(
    "--temperature",
    type=float,
    help="Temperature for generating explanation (between 0 and 1)",
    default=0
)

parser.add_argument(
    "--top_p",
    type=float,
    help="Nucleus sampling paramter for generating explanation",
    default=1.0
)

parser.add_argument(
    "--num_return_sequences",
    type=int,
    help="No of explanations to generate",
    default=1
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
    "--rationalize",
    type=str,
    help="Set to True for rationalization",
    choices=["True", "False"],
    default="False"
)

args = parser.parse_args()
assert args.config.endswith(".json")
with open(args.config, "r") as f:
    parser.set_defaults(**json.load(f))
#---------------------------------------------------------- 
def main():
    args = parser.parse_args()

    if args.shuffle == "True":
        args.shuffle = True
    else: 
        args.shuffle = False

    if args.do_sample == "True":
        args.do_sample = True
    else: 
        args.do_sample = False

    if args.rationalize == "True":
        args.rationalize = True
    else: 
        args.rationalize = False

    assert args.output_file != None
    assert args.output_file.endswith(".json")
    check_if_exists("/".join(args.output_file.split("/")[:-1]), isDir=True, createIfNotExists=True)

    set_seed(args.seed)

    logging.basicConfig(filemode='w', level=logging.INFO)

    logging.info("Args:")
    logging.info('\n\t'.join(f'{k}={v}' for k, v in vars(args).items()))

    #Assumption: Explain model uses the same tokenizer as the direct model
    direct_model_path = args.direct_model_name if args.direct_model_name != None else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(direct_model_path, cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    tokenizer.model_max_length = args.model_max_length
    max_length = args.max_length
    if max_length == -1:
        max_length = tokenizer.model_max_length

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
        save_to_disk=False,
        disk_path=args.out_path
    )

    data_loader = create_data_loader(
        ds, 
        args.batch_size,
        args.shuffle
    )

    device = "cpu"
    if torch.cuda.is_available:
        device = "cuda"

    direct_model = LlamaForCausalLM.from_pretrained(
        direct_model_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        cache_dir=args.cache_dir,
        attn_implementation="flash_attention_2",
    )
    logging.info("Loaded direct model {} on device {}".format(direct_model_path, device))

    direct_model = direct_model.eval()
    direct_model.generation_config.pad_token_id = tokenizer.pad_token_id
    direct_model.resize_token_embeddings(len(tokenizer))

    if args.explain_model_name != None:
        explain_model = LlamaForCausalLM.from_pretrained(
            args.explain_model_name,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            cache_dir=args.cache_dir,
            attn_implementation="flash_attention_2",
        )
        logging.info("Loaded explain model {} on device {}".format(args.explain_model_name, device))
        explain_model = explain_model.eval()
        explain_model.generation_config.pad_token_id = tokenizer.pad_token_id
        explain_model.resize_token_embeddings(len(tokenizer))
    else: 
        explain_model = direct_model
        logging.info("Loaded explain model {} on device {}".format(direct_model_path, device))

    failed = 0
    passed = 0
    nlls_before=[]
    nlls_after=[]
    victory, defeat = 0, 0
    outcome_margin = []
    victory_margin, defeat_margin = [], []
    stage_1_time, stage_2_time, stage_3_time = [], [], []    
    outcome = []
    entropy_before, entropy_after = [], []
    logging.info("Max Entropy: {}".format(torch.log(torch.tensor(tokenizer.vocab_size)).item()))
    
    is_instruct=("instruct" in (direct_model_path).lower())

    nlls_before_max = -1
    instances_to_save = {
        "args": '\n\t'.join(f'{k}={v}' for k, v in vars(args).items()), 
        "instances": []
    }
    acc_direct, acc_explain = [], []
    promptLlama3 = PromptLlama3()
    with torch.no_grad():
        assert args.batch_size==1
        for id, text, chunk_text, start, end in tqdm(data_loader, desc="Data loader"):                         
            if len(start[0]) == 0:
                logging.info("Not enough tokens in current text! Skipping...")
                continue
            for i, (token_start, token_end) in tqdm(enumerate(zip(start[0], end[0])), desc="Text"):
                if args.target_len >= 0 and i > args.target_len:
                    break

                input_text = text[0][:token_start]
                input_text_direct = input_text_explain = input_text
                
                next_token = text[0][token_start:token_end]
                if len(next_token.strip()) == 0:
                    #Do not skip end of output
                    if token_start == len(text[0]): #End of sequence token
                        next_token = next_token_direct = next_token_explain  = "<|eot_id|>"
                    else: #Erroneous token, skip
                        continue
                else: #Do not skip <|eot_id|> instance
                    #To increase data diversity
                    #Do not skip start of output
                    if token_start > 0 and np.random.rand() < args.skip_p:
                        continue
                next_token_direct = next_token
                next_token_explain = next_token
                    
                next_token_ids_direct = tokenizer(
                    next_token_direct, 
                    add_special_tokens=False
                )["input_ids"]
                next_token_ids_direct = torch.tensor(next_token_ids_direct)

                next_token_ids_explain = tokenizer(
                    next_token_explain, 
                    add_special_tokens=False
                )["input_ids"]
                next_token_ids_explain = torch.tensor(next_token_ids_explain)

                prompt_direct = promptLlama3.get_prompt(
                    instance={
                        "text": input_text_direct,
                        "next_token": next_token_direct,
                    }, 
                    is_instruct=is_instruct, 
                    granularity_level=args.granularity_level, 
                    prompt_type="direct", 
                    return_messages=False, 
                    omit_last_body=False, 
                    rationalize=False
                )

                input_ids_direct = tokenizer.encode(
                    prompt_direct, 
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                target_ids_direct = input_ids_direct.clone()
                try:
                    target_ids_direct = mark_target_tokens(
                        next_token_ids=next_token_ids_direct, 
                        target_ids=target_ids_direct, 
                        tokenizer=tokenizer,
                        setting="direct"
                    )
                    #Stage 1: Get the direct completion
                    start_time = time.time()
                    out = direct_model(
                        input_ids=input_ids_direct,
                        labels=target_ids_direct
                    )
                    end_time = time.time()
                    logging.info("Stage 1: Time elapsed: {:0.5f} seconds".format(end_time-start_time))
                    stage_1_time.append((end_time-start_time))

                    direct_entropy, next_token_prediction_direct, target_rank_direct = get_entropy(
                        logits=out["logits"],
                        target_ids=target_ids_direct,
                        tokenizer=tokenizer
                    )

                    nlls_before.append(out.loss)
                    entropy_before.append(direct_entropy.mean().item())

                    rationalize=False
                    while 1:
                        if rationalize:
                            logging.info("Rationalizing...")
                        #Stage 2: Get the explanation
                        start_time = time.time()
                        
                        prompt_explain = promptLlama3.get_prompt(
                            instance={
                                "text": input_text_explain,
                                "next_token": next_token_explain,
                            }, 
                            is_instruct=is_instruct, 
                            granularity_level=args.granularity_level, 
                            prompt_type="generate_explain", 
                            return_messages=False, 
                            omit_last_body=True, 
                            rationalize=rationalize
                        )

                        input_ids_explain = tokenizer.encode(
                            prompt_explain, 
                            max_length=max_length,
                            truncation=True,
                            return_tensors="pt"
                        )
                        generation_outputs = explain_model.generate(
                            input_ids=input_ids_explain.to(device),
                            max_new_tokens=args.max_new_tokens,
                            num_beams=args.num_beams,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_return_sequences=args.num_return_sequences
                        )
                        end_time = time.time()
                        logging.info("Stage 2: Time elapsed: {:0.5f} seconds".format(end_time-start_time))
                        if not rationalize:
                            stage_2_time.append((end_time-start_time))

                        #START: Choose best explanation
                        response = ""
                        cur_explain_response = ""
                        out = ""
                        prompt_answer = ""
                        prompt_answer_input_ids = None
                        prompt_answer_target_ids = None
                        best_loss = torch.inf
                        start_time = 0
                        end_time = 0
                        for gen_i, generation_output in tqdm(enumerate(generation_outputs), desc="num_return_sequences"):
                            if gen_i >= args.num_return_sequences:
                                break
                            cur_response = tokenizer.decode(generation_output[len(input_ids_explain[0]):], skip_special_tokens=True)

                            #START: For logging purpose
                            cur_explain_response = cur_response.strip()
                            #END: For logging purpose

                            cur_response = parse_response(cur_response)

                            #Stage 3: Get the final completion
                            cur_start_time = time.time()
                            cur_prompt_answer = promptLlama3.get_prompt(
                                instance={
                                    "text": input_text_explain,
                                    "explanation": cur_response,
                                    "next_token": next_token_explain,
                                }, 
                                is_instruct=is_instruct, 
                                granularity_level=args.granularity_level, 
                                prompt_type="explain_answer", 
                                return_messages=False, 
                                omit_last_body=False, 
                                rationalize=False
                            )

                            cur_prompt_answer_input_ids = tokenizer.encode(
                                cur_prompt_answer,
                                max_length=max_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            cur_prompt_answer_target_ids = cur_prompt_answer_input_ids.clone()
                            cur_prompt_answer_target_ids = mark_target_tokens(
                                next_token_ids=next_token_ids_explain, 
                                target_ids=cur_prompt_answer_target_ids, 
                                tokenizer=tokenizer,
                                setting="answer"
                            )

                            cur_out = explain_model(
                                input_ids=cur_prompt_answer_input_ids,
                                labels=cur_prompt_answer_target_ids
                            )
                            cur_end_time = time.time()
                            if cur_out.loss.item() <= best_loss:
                                best_loss = cur_out.loss.item()
                                out = cur_out
                                response = cur_response 
                                explain_response = cur_explain_response
                                prompt_answer = cur_prompt_answer
                                prompt_answer_input_ids = cur_prompt_answer_input_ids
                                prompt_answer_target_ids = cur_prompt_answer_target_ids
                                start_time = cur_start_time
                                end_time = cur_end_time
                        #END: Choose best explanation   
                        logging.info("Stage 3: Time elapsed: {:0.5f} seconds".format(end_time-start_time))
                        if not rationalize:
                            stage_3_time.append((end_time-start_time))
                        if rationalize or not args.rationalize:
                            break
                        if out.loss < nlls_before[-1]:
                            break
                        rationalize = True

                    explanation_entropy, next_token_prediction_explain, target_rank_explain = get_entropy(
                            logits=out["logits"],
                            target_ids=prompt_answer_target_ids,
                            tokenizer=tokenizer
                        )
                    nlls_after.append(out.loss)
                    entropy_after.append(explanation_entropy.mean().item())

                    outcome_margin.append((nlls_before[-1]-nlls_after[-1]))
                    if nlls_after[-1] < nlls_before[-1]:
                        victory +=1
                        victory_margin.append(abs(nlls_after[-1] - nlls_before[-1]))
                        outcome.append(1)
                    else: 
                        defeat += 1
                        defeat_margin.append(abs(nlls_after[-1] - nlls_before[-1]))
                        outcome.append(0)

                    if next_token_prediction_direct.strip() == next_token.strip():
                        acc_direct.append(1)
                    else: 
                        acc_direct.append(0)

                    if next_token_prediction_explain.strip() == next_token.strip():
                        acc_explain.append(1)
                    else: 
                        acc_explain.append(0)

                    instances_to_save["instances"].append({
                        "id": id,
                        "text": text,
                        "token_ind": i,
                        "token_start": token_start,
                        "token_end": token_end,
                        "prompt_direct": prompt_direct,
                        "prompt_explain": prompt_explain, 
                        "prompt_answer": prompt_answer,
                        "explanation": explain_response,
                        "next_token": next_token,
                        "next_token_explain": next_token_explain,
                        "outcome": outcome[-1],
                        "outcome_margin": outcome_margin[-1].item(),
                        "loss_direct": nlls_before[-1].item(),
                        "loss_explain": nlls_after[-1].item(),
                        "entropy_direct": entropy_before[-1],
                        "entropy_explain": entropy_after[-1],
                        "accuracy_direct": acc_direct[-1],
                        "accuracy_explain": acc_explain[-1],
                        "target_rank_direct": target_rank_direct,
                        "target_rank_explain": target_rank_explain,
                    })

                    logging.info("Direct prompt:\n{}\n******".format(prompt_direct))
                    logging.info("Answer prompt:\n{}\n******".format(prompt_answer))
                    logging.info("Explain prompt:\n{}{}".format(prompt_explain, explain_response))
                    logging.info("Direct prediction: {}".format(next_token_prediction_direct.strip()))
                    logging.info("\tDirect rank: {:0.0f}".format(1+target_rank_direct))
                    logging.info("Explain prediction: {}".format(next_token_prediction_explain.strip()))
                    logging.info("\tExplain rank: {:0.0f}".format(1+target_rank_explain))
                    logging.info("next_token: {}".format(next_token.strip()))
                    logging.info("Before: Loss: {:0.2f}, Entropy: {:0.2f}".format(nlls_before[-1], entropy_before[-1]))
                    logging.info("After:  Loss: {:0.2f}, Entropy: {:0.2f}".format(nlls_after[-1], entropy_after[-1]))
                    
                    ppl_before = torch.exp(torch.stack(nlls_before).mean())
                    ppl_after = torch.exp(torch.stack(nlls_after).mean())
                    logging.info("Perplexity (Before): {}".format(ppl_before))
                    logging.info("Perplexity (After): {}".format(ppl_after))
                    if outcome[-1] == 1:
                        logging.info("\nExplanation beat Direct prompting.")
                    else: 
                        logging.info("\nDirect prompting beat Explanation.")
                    logging.info("Win rate:\n\tExpl: {:0.2f}%".format(((victory)/(victory+defeat))*100))
                    logging.info("Accuracy:\n\tDir: {:0.2f}%\n\tExpl: {:0.2f}%".format(((sum(acc_direct))/len(acc_direct))*100, ((sum(acc_explain))/len(acc_explain))*100))

                    nlls_before_max = max(nlls_before_max, nlls_before[-1])
                    MAX_LOSS = min(2.5, nlls_before_max)
                    reasonable_loss = torch.where(torch.stack(nlls_after)<MAX_LOSS)
                    if len(reasonable_loss[0]):
                        reasonable_ppl = torch.exp(torch.stack(nlls_after)[reasonable_loss].mean())
                        reasonable_ppl_before = torch.exp(torch.stack(nlls_before)[reasonable_loss].mean())
                        logging.info("When After: Loss < {} in {}/{} ({:0.2f}%) instances".format(MAX_LOSS, len(reasonable_loss[0]), len(nlls_after), 100*(len(reasonable_loss[0])/len(nlls_after))))
                        logging.info("\tPerplexity (Before): {}".format(reasonable_ppl_before))
                        logging.info("\tPerplexity (After): {}".format(reasonable_ppl))

                        reasonable_outcome_margin = np.array(outcome_margin)[reasonable_loss[0]]
                        positive_inds = np.where(reasonable_outcome_margin >= 0)
                        negative_inds = np.where(reasonable_outcome_margin < 0)
                        reasonable_victory_margin = []
                        reasonable_defeat_margin = []
                        if len(positive_inds):
                            reasonable_victory_margin = reasonable_outcome_margin[positive_inds]
                        if len(negative_inds):
                            reasonable_defeat_margin = -reasonable_outcome_margin[negative_inds]
                        if len(reasonable_victory_margin):
                            logging.info("\tAvg victory margin: {:0.2f}, min: {}, max: {}".format(sum(reasonable_victory_margin)/len(reasonable_victory_margin), min(reasonable_victory_margin), max(reasonable_victory_margin)))
                        if len(reasonable_defeat_margin):
                            logging.info("\tAvg defeat margin: {:0.2f}, min: {}, max: {}".format(sum(reasonable_defeat_margin)/len(reasonable_defeat_margin), min(reasonable_defeat_margin), max(reasonable_defeat_margin)))
                        
                        reasonable_outcome = np.array(outcome)[reasonable_loss]
                        logging.info("\tWin rate:\n\t\tExpl: {:0.2f}%".format(((reasonable_outcome.sum()))/(len(reasonable_outcome))*100))
                    logging.info("*"*20)
                    
                    passed += 1
                    clear_memory(
                        prompt_direct, 
                        prompt_explain, 
                        prompt_answer,
                        input_ids_direct,
                        target_ids_direct,
                        input_ids_explain,
                        generation_outputs,
                        response,
                        prompt_answer_input_ids
                    )
                except Exception as error: 
                    logging.info("Error {}: {}".format(type(error).__name__, error))
                    logging.info("Could not process input: {}".format(len(input_ids_direct[0])))
                    failed += 1
                    continue
                if passed % 100 == 0:
                    with open(args.output_file, "w") as f: 
                        json.dump(instances_to_save, f)
                    if len(victory_margin):
                        logging.info("Avg victory margin: {:0.2f}, min: {}, max: {}".format(sum(victory_margin)/len(victory_margin), min(victory_margin), max(victory_margin)))
                    if len(defeat_margin):
                        logging.info("Avg defeat margin: {:0.2f}, min: {}, max: {}".format(sum(defeat_margin)/len(defeat_margin), min(defeat_margin), max(defeat_margin)))
                    logging.info("Avg stage 1 time: {:0.2f} seconds".format(sum(stage_1_time)/len(stage_1_time)))
                    logging.info("Avg stage 2 time: {:0.2f} seconds".format(sum(stage_2_time)/len(stage_2_time)))
                    logging.info("Avg stage 3 time: {:0.2f} seconds".format(sum(stage_3_time)/len(stage_3_time)))
                    logging.info("Processed {}/{} ({:0.2f}%)".format(passed, passed+failed, 100*(passed/(passed+failed))))
    with open(args.output_file, "w") as f: 
        json.dump(instances_to_save, f)
    if len(victory_margin):
        logging.info("Avg victory margin: {:0.2f}, min: {}, max: {}".format(sum(victory_margin)/len(victory_margin), min(victory_margin), max(victory_margin)))
    if len(defeat_margin):
        logging.info("Avg defeat margin: {:0.2f}, min: {}, max: {}".format(sum(defeat_margin)/len(defeat_margin), min(defeat_margin), max(defeat_margin)))
    logging.info("Avg stage 1 time: {:0.2f} seconds".format(sum(stage_1_time)/len(stage_1_time)))
    logging.info("Avg stage 2 time: {:0.2f} seconds".format(sum(stage_2_time)/len(stage_2_time)))
    logging.info("Avg stage 3 time: {:0.2f} seconds".format(sum(stage_3_time)/len(stage_3_time)))
    logging.info("Processed {}/{} ({:0.2f}%)".format(passed, passed+failed, 100*(passed/(passed+failed))))
#----------------------------------------------------------
if __name__ == "__main__":
    main()
