import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from FoundationModelsProject2.models import TransformerLanguageModel
from FoundationModelsProject2.data_manipulation import TextDataset, collate_fn
from FoundationModelsProject2.tokenizer import load_tokenizer
from FoundationModelsProject2.model_training import train_model, ModelEvaluator

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
EMBED_DIM = 512
NUM_LAYERS = 3
NUM_HEADS = 4
VOCAB_SIZE = 10000
LEARNING_RATE = 1e-3
EPOCHS = 30
BATCH_SIZE = 128
TRAIN_FILE = "../data/split_data/train.jsonl"
VAL_FILE = "../data/split_data/val.jsonl"
MAX_SEQ_LEN = 512
TOKENIZER_PATH = "../tokenizer/bpe_tokenizer.model"
MODEL_PATH = "../model_training/saved_models/transformer.pth"

model = TransformerLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
)

train_losses, val_losses = None, None
# Load model if it already exists, otherwise train model and save
if os.path.exists(MODEL_PATH) and True:
    print("Loading model...")
    model = model.to(DEVICE)
    loaded_model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(loaded_model)
else:
    train_losses, val_losses, model = train_model(device=DEVICE, model=model, tokenizer_path=TOKENIZER_PATH, train_file_path=TRAIN_FILE,val_file_path=VAL_FILE, max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS)
    torch.save(model.state_dict(), MODEL_PATH)

model_eval = ModelEvaluator(model)

# Generate loss curve if model was freshly trained
if train_losses is not None:
    model_eval.gen_loss_curve_plot(train_losses, val_losses, "../model_training/loss_plots/transformer_loss_curve.png")

# Load test dataset and pass to different evaluation functions
tokenizer = load_tokenizer(TOKENIZER_PATH)
test_dataset = TextDataset("../data/test.jsonl", tokenizer, MAX_SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Calculate and print model perplexity
test_perplexity = model_eval.evaluate_perplexity(test_loader, nn.CrossEntropyLoss(ignore_index=3), DEVICE, 10000)
print(f"Test Set Perplexity: {test_perplexity:.2f}")
# Calculate and print model BLEU score
test_bleu_score = model_eval.evaluate_bleu(test_loader, DEVICE, tokenizer, MAX_SEQ_LEN)
print(f"Test Set BLEU Score: {test_bleu_score:.8f}")
# Calculate and print model loss
test_loss = model_eval.evaluate_loss(test_loader, DEVICE, nn.CrossEntropyLoss(ignore_index=3))
print(f"Test Loss: {test_loss:.4f}")

# Test model generative output responses to 2 test prompts
prompt_1 = "Which do you prefer? Dogs or cats?"
prompt_2 = "What do you know about top hats?"

test_response_1 = model.generate(tokenizer=tokenizer, prompt=prompt_1, max_length=MAX_SEQ_LEN, eos_token_id=2, device=DEVICE)
print(f"Response 1: {test_response_1}")
test_response_2 = model.generate(tokenizer=tokenizer, prompt=prompt_2, max_length=MAX_SEQ_LEN, eos_token_id=2, device=DEVICE)
print(f"Response 2: {test_response_2}")