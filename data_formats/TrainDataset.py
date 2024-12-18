from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class TrainDataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length):
        self.prompts = prompts 
        assert tokenizer.model_max_length >= max_length, "max_length ({}) cannot be longer than model_max_length ({})".format(max_length, tokenizer.model_max_length)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_value = tokenizer.pad_token_id

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        cur_prompt = self.prompts[idx]
        input_ids = self.tokenizer.encode(
            cur_prompt,
            padding="do_not_pad",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).squeeze(0)
        labels = input_ids.clone()
        return input_ids, labels

    def collate_batch(self, batch):
        input_ids, labels = zip(*batch)
        max_len = max([t.size(0) for t in input_ids])
        input_ids = torch.stack([F.pad(t, (0, max_len - t.size(0)), value=self.padding_value) for t in input_ids])
        labels = torch.stack([F.pad(t, (0, max_len - t.size(0)), value=self.padding_value) for t in labels])
        return {
            "input_ids": input_ids,
            "labels": labels
        }