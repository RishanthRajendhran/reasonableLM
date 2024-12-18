from datasets import load_dataset, load_from_disk
from typing import Callable, List, Dict, Any, Tuple
from datasets import Dataset, IterableDataset
import random
import logging

class RedPajama:
    def __init__(self, path:str, chunker:Callable[[str, str, int], List[Dict[str, Any]]], skip_first_n_tokens:int=0, granularity_level:str="sentence", granularity_len:int=-1):
        self.path = path
        self.data = None
        self.chunker = chunker
        self.skip_first_n_tokens = skip_first_n_tokens
        self.granularity_level = granularity_level
        self.granularity_len = granularity_len

    def save_to_disk(self, disk_path):
        self.data.save_to_disk(disk_path)

    def __len__(self):
        return len(self.data)

    def load_dataset(self, split:str="train", num_examples:int=100, cache_dir:str="/huggingface_cache/datasets/", streaming:bool=False, shuffle:bool=False, from_disk:str=None)->None:
        if from_disk != None:
            self.data = load_from_disk(from_disk)
        else:
            self.data = load_dataset(
                self.path,
                split=split,
                cache_dir=cache_dir,
                streaming=streaming
            )
            if shuffle:
                self.data = self.data.shuffle(buffer_size=100*num_examples)
            self.data = self.data.take(num_examples)
            if isinstance(self.data, IterableDataset):
                data_list = [example for example in self.data]
                self.data = Dataset.from_dict({key: [d[key] for d in data_list] for key in data_list[0]})

    def apply_map(self, map_func:Callable[[Dict[str, Any]],Dict[str, Any]], remove_columns:bool=False):
        self.data = self.data.map(
            map_func, 
            remove_columns=[col for col in self.data.column_names if col!="__index_level_0__"] if remove_columns else None
        )
    
    def format_instance(self, instance, is_inference:bool=False)->str:
        formatted_instance = instance["text"]
        return formatted_instance

    def __getitem__(self, idx:int)->Tuple[str, List[str], List[int], List[int]]:
        item = self.data[idx]
        #Use prefix text for input in case of instruction-following datasets
        prefix_text = ""
        #Chunk only the response text
        chunked_text = self.chunker(item["text"], self.skip_first_n_tokens, self.granularity_level, self.granularity_len)
        out = {
            "text": [],
            "start": [],
            "end": [],
            "id": [],
        }
        for chunk in chunked_text:
            out["text"].append(self.format_instance({
                "text": chunk["text"]
            }))
            out["start"].append(len(prefix_text)+chunk["start"])
            out["end"].append(len(prefix_text)+chunk["end"])
        full_text = self.format_instance({
            "text": item["text"]
        })
        id = item["__index_level_0__"]
        return id, full_text, out["text"], out["start"], out["end"]
    
    def collate_batch(self, batch:Tuple[List[Any], List[Any], List[Any], List[Any]])->Tuple[List[Any], List[Any], List[Any], List[Any], List[Any]]:
        id, full_text, chunk_text, start, end = zip(*batch)
        return id, full_text, chunk_text, start, end