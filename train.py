import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AdamW, get_scheduler
from tqdm import tqdm
import argparse
import json
import logging
from utils.data_loader import create_train_data_loader
from utils.misc import check_if_exists

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--config", 
    type=str,
    help="Path to json config file"
    # required=True
)

parser.add_argument(
    "--model_name",
    type=str,
    help="Path to HF Llama model",
    # required=True
)

parser.add_argument(
    "--cache_dir",
    type=str,
    help="Path to HF cache",
    # required=True
)

parser.add_argument(
    "--max_length",
    type=int,
    help="Max length of prompt (in no. of tokens)",
    default=-1
)

parser.add_argument(
    "--shuffle",
    help="Shuffle dataset",
    default="False"
)

parser.add_argument(
    "--model_max_length",
    type=int,
    help="Model max context length",
    default=128000
)

parser.add_argument(
    "--num_epochs",
    type=int,
    help="Number of training epochs",
    default=1
)

parser.add_argument(
    "--data_path",
    type=str,
    help="Path to data containing instances",
    default=None
)

parser.add_argument(
    "--out_model_path",
    type=str,
    help="Path to save trained model",
    default="./model_outputs/model.pt"
)

parser.add_argument(
    "-learning_rate",
    type=float,
    help="Learning rate for training",
    default=3e-5
)

parser.add_argument(
    "--batch_size",
    type=int,
    help="Batch size for training",
    default=1
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

    assert args.data_path != None
    check_if_exists("/".join(args.out_model_path.split("/")[:-1]), isDir=True, createIfNotExists=True)

    logging.basicConfig(filemode='w', level=logging.INFO)

    logging.info("Args:")
    logging.info('\n\t'.join(f'{k}={v}' for k, v in vars(args).items()))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    tokenizer.model_max_length = args.model_max_length
    max_length = args.max_length
    if max_length == -1:
        max_length = tokenizer.model_max_length

    train_dataloader = create_train_data_loader(args.data_path, tokenizer, max_length, args.batch_size, args.shuffle)

    device = "cpu"
    if torch.cuda.is_available:
        device = "cuda"

    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        cache_dir=args.cache_dir,
        attn_implementation="flash_attention_2",
    )
    logging.info("Loaded model {} on device {}".format(args.model_name, device))

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(args.num_epochs):
        train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.item()
            logging.info("Training Loss: {}".format(loss.item()))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        if train_loss == 0:
            break
    model.save_pretrained(args.out_model_path)
    tokenizer.save_pretrained(args.out_model_path)
#----------------------------------------------------------
if __name__ == "__main__":
    main()
