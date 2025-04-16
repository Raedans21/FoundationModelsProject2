import os
import sentencepiece as spm

# Merge all text files into a single corpus
def merge_text_files(data_dir, output_file):
    """
    This will merge all textual data in a directory into a single corpus
    :param data_dir: path to the director containing the raw text files
    :param output_file: path to the file where corpus will be saved
    """
    with open(output_file, 'w', encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding="utf-8") as infile:
                    outfile.write(infile.read())

if __name__ == '__main__':
    DATA_DIR = "../data/raw"  # path to raw data directory
    TOKENIZER_PREFIX = "bpe_tokenizer"
    VOCAB_SIZE = 10000 # tokenizer stop condition
    CORPUS_FILE = "../corpus.txt"  # path to new combined corpus file
    merge_text_files(DATA_DIR, CORPUS_FILE)

    # Train the tokenizer with special tokens
    spm.SentencePieceTrainer.Train(
        input=CORPUS_FILE,
        model_prefix=TOKENIZER_PREFIX,
        model_type="bpe",
        vocab_size=VOCAB_SIZE,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=",".join(["<bos>", "<eos>", "<pad>"])
    )

    print("Tokenizer training complete! Files generated:")
    print(f"- {TOKENIZER_PREFIX}.model")
    print(f"- {TOKENIZER_PREFIX}.vocab")
