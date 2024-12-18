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
    help="Path to directory containing batch_id and save batch responses.",
)

parser.add_argument(
    "--batch_id",
    type=str,
    help="OpenAI batch ID",
    default=None
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

    batch_id = args.batch_id
    if batch_id==None:
        with open(args.batch_path+"/"+"openai_batch.pkl", "rb") as f:
            batch_id=pkl.load(f)["response"].id

    client = openai.OpenAI(api_key=args.openai_api_key)

    response = client.batches.retrieve(batch_id)

    logging.info("Status: {}".format(response))
    if response.status == "completed":
        file_response = client.files.content(response.output_file_id)

        with open(args.batch_path+"/"+"openai_batch_responses.jsonl","w") as f:
            f.write(file_response.text)
#----------------------------------------------------------
if __name__ == "__main__":
    main()