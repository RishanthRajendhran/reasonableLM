from data_formats import RedPajama, TrainDataset
from torch.utils.data import DataLoader
import logging
from utils.misc import read_file
import pandas as pd
#---------------------------------------------------------------------------
def create_dataset(
    dataset, 
    data_path, 
    chunker, 
    skip_first_n_tokens,
    granularity_level, 
    granularity_len,
    num_examples=100, 
    split="train", 
    cache_dir="/huggingface_cache/datasets",
    streaming=False,
    shuffle=False, 
    load_from_disk=False,
    save_to_disk=False,
    disk_path=None
):
    if dataset == "redpajama":
        DataClass = RedPajama.RedPajama
    else: 
        raise ValueError("Unrecognized dataset: {}".format(dataset))

    ds = DataClass(
        data_path,  
        chunker.chunk, 
        skip_first_n_tokens, 
        granularity_level, 
        granularity_len
    )

    ds.load_dataset(
        split, 
        num_examples,
        cache_dir,
        streaming=streaming,
        shuffle=shuffle,
        from_disk=disk_path if load_from_disk else None
    )

    if save_to_disk:
        assert disk_path!=None, "disk_path not specified!" 
        ds.save_to_disk(disk_path)

    return ds
#---------------------------------------------------------------------------
def create_train_data_loader(data_path, tokenizer, max_length, batch_size, shuffle):
    data = read_file(data_path)
    instances = data["instances"]
    df = pd.DataFrame(instances)
    df_explain = df[df["outcome"] == 1]
    df_explain = df[df["accuracy_explain"] == 1]
    # df_direct = df[df["accuracy_direct"] == 1]
    # df_direct = df_direct.sample(n=min(len(df_explain), len(df_direct)))
    prompts = []
    prompts.extend(df_explain["prompt_answer"].tolist())
    prompts.extend(df_explain["prompt_direct"].tolist())

    ds = TrainDataset.TrainDataset(
        prompts,
        tokenizer, 
        max_length,
    )

    return create_data_loader(ds, batch_size, shuffle)
#---------------------------------------------------------------------------
def create_data_loader(
    ds,
    batch_size,
    shuffle
):
    
    # if shuffle:
    #     logging.warning("Usage deprecated: shuffle")

    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        collate_fn=ds.collate_batch
    )
    return data_loader
#---------------------------------------------------------------------------