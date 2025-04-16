import torch.nn as nn

def collate_fn(batch):
    """
    Ensure batch is appropriately sized and padded for efficient training
    :param batch: batch from DataLoader, which will be a list of Tuples of token ID tensors
        (which could be different sizes)
    :return: collated input and target batch
    """
    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=3)
    target_batch = nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=3)
    return input_batch, target_batch