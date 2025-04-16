import torch
from torch.utils.data import Dataset
import json
from typing import Tuple

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_seq_length=128):
        """
        Create a text dataset for PyTorch Dataset that handles our jsonl prompts+completions
        for Causal LM
        :param filepath: path to the jsonl file
        :param tokenizer: instance of trained SentencePiece tokenizer
        :param max_seq_length: maximum sequence length that is allowed
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        prompts = []
        completions = []

        # open the jsonl file and tokenize each sample
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompts.append(item.get("prompt", ""))
                completions.append(item.get("completion", ""))

        prompts, completions = add_special_tokens((prompts, completions))

        for prompt, completion in zip(prompts, completions):
            # Causal Language Modeling, thus prompts and completion are treated the same way
            text = prompt + " " + completion
            # Tokenize the full prompt + completion (truncate at max sequence length)
            token_ids = tokenizer.encode(text, out_type=int)[:max_seq_length]
            # Avoid overly short samples
            if len(token_ids) < 2:
                continue
            # Append tokenized sample to list
            self.samples.append(token_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get and format a sample at the given index
        For Causal Language Modeling, the model will be trained to predict every next token in the sequence
        given the prior ones
        :param idx:
        :return:
        """
        tokens = self.samples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids

def add_special_tokens(pairs: Tuple[list, list]):
    """
    Insert <bos> and <eos> special tokens into a dataset
    :param pairs: original prompts and completions
    :return: prompts/completion pairs with special tokens inserts
    """
    prompts, completions = pairs
    new_prompts = []
    new_completions = []

    for prompt, completion in zip(prompts, completions):
        # If the beginning of the prompt is upper case, then we assume it is the start of a sequence
        if prompt[0].isupper():
            prompt = '<bos>' + prompt
        # If the end of the completion is a termination punctuation, then we assume it is the end of a sequence
        if completion.endswith('.') or completion.endswith('?') or completion.endswith('!'):
            completion += '<eos>'
        new_prompts.append(prompt)
        new_completions.append(completion)

    return new_prompts, new_completions