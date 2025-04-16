import sentencepiece as spm

def load_tokenizer(tokenizer_path):
    """
    Initialize and load in tokenizer from tokenizer file path
    :param tokenizer_path: location of tokenizer file
    :return: loaded tokenizer object
    """
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    return tokenizer