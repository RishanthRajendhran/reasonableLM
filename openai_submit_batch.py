import pickle as pkl
from tqdm import tqdm
import argparse
import json
import logging
import openai

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--config", 
    type=str,
    help="Path to json config file"
    # required=True
)

parser.add_argument(
    "--cache_dir",
    type=str,
    help="Path to HF cache",
)

parser.add_argument(
    "--batch_path",
    type=str,
    help="Path to directory to containing batches. This is also where batch_id gets saved.",
)

parser.add_argument(
    "--openai_api_key",
    type=str,
    help="OpenAI Secret Key"
)

args = parser.parse_args()
assert args.config.endswith(".json")
with open(args.config, "r") as f:
    parser.set_defaults(**json.load(f))
#---------------------------------------------------------- 
def main():
    args = parser.parse_args()

    logging.basicConfig(filemode='w', level=logging.INFO)

    logging.info("Args:")
    logging.info('\n\t'.join(f'{k}={v}' for k, v in vars(args).items()))

    client = openai.OpenAI(api_key=args.openai_api_key)

    batch_input_file = client.files.create(
        file=open(args.batch_path+"/"+"openai_batch.jsonl", "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "ReasonableLM"
        }
    )
    logging.info("batch_input_file_id: {}".format(batch_input_file_id))
    logging.info("response:\n{}".format(response))

    with open(args.batch_path+"/"+"openai_batch.pkl", "wb") as f:
        pkl.dump({
            "batch_id": batch_input_file_id,
            "response": response,
        }, f)
#----------------------------------------------------------
if __name__ == "__main__":
    main()