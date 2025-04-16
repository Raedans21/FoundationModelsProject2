import torch
import random
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, model):
        """
        Create a unified evaluator class for the given model to consolidate evaluation methods in one place
        :param model: model being evaluated
        """
        self.model = model

    def evaluate_perplexity(self, data_loader, criterion, device, vocab_size):
        """
        Evaluate model perplexity on the given dataset
        :param data_loader: PyTorch Dataloader for the dataset
        :param criterion: model loss criterion
        :param device: device to load the model and inputs to
        :param vocab_size: tokenizer vocabulary size
        :return: calculated perplexity value
        """
        self.model.eval()
        # Calculate average test set loss for usage in perplexity calculation
        total_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in data_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                model_outputs = self.model(input_ids)
                # Handle different model output formats - logits and hidden state versus just logits
                if isinstance(model_outputs, tuple):
                    logits, _ = model_outputs
                else:
                    logits = model_outputs
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)

        perplexity = torch.exp(torch.tensor(avg_loss))
        return perplexity.item()

    def evaluate_bleu(self, data_loader, device, tokenizer, max_seq_len, temperature=1.0):
        """
        Evaluate model Bilingual Evaluation Understudy (BLEU) on the given dataset
        :param data_loader: PyTorch Dataloader for the dataset
        :param device: device to load the model and inputs to
        :param tokenizer: trained tokenizer object
        :param max_seq_len: maximum sequence length
        :param temperature: model generation temperature
        :return: calculated BLEU score
        """
        self.model.eval()
        # Store all predictions and references to feed to bleu score calculator
        predictions = []
        targets = []

        with torch.no_grad():
            for input_ids, target_ids in tqdm(data_loader, desc=f"BLEU Score Eval"):
                input_ids = input_ids.to(device)

                # Iterate through each sample within a batch
                for i in range(input_ids.size(0)):
                    prompt = tokenizer.decode(input_ids[i].tolist(), out_type=str)
                    generated_text = self.model.generate(tokenizer=tokenizer, prompt=prompt, max_length=max_seq_len, eos_token_id=2, temperature=temperature, device=device)

                    # Convert text to a list of token id's
                    pred_token_ids = tokenizer.encode(generated_text, out_type=int)
                    target_token_ids = target_ids[i].tolist()

                    # Use token id lists to convert each token id to its respective token string
                    pred_tokens = [tokenizer.id_to_piece(token_id) for token_id in pred_token_ids]
                    target_tokens = [tokenizer.id_to_piece(token_id) for token_id in target_token_ids]

                    predictions.append(pred_tokens)
                    targets.append([target_tokens])

        # NLTK smoothing function to prevent non-present n-grams from zeroing out BLEU score
        smoothing_func = SmoothingFunction().method1
        score = corpus_bleu(list_of_references=targets, hypotheses=predictions, smoothing_function=smoothing_func)
        return score

    def evaluate_loss(self, data_loader, device, criterion):
        """
        Evaluate model loss on the given dataset
        :param data_loader: Dataloader for the dataset
        :param device: Device to load the model and inputs to
        :param criterion: Loss criterion
        :return: Averaged model loss on the given dataset
        """
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for input_ids, target_ids in data_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                outputs = self.model(input_ids)
                # Handle different model output formats - logits and hidden state versus just logits
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs

                # Flatten the outputs and targets to compute loss over all tokens
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches
        return avg_loss

    def gen_loss_curve_plot(self, train_losses, val_losses, file_path=None):
        """
        Generates and saves a plot of training and validation loss curves
        :param train_losses: array of training loss values at each epoch
        :param val_losses: array of validation loss values at each epoch
        :param file_path: file path to save the plot to
        """
        # Plot the training and validation curves
        plt.plot(train_losses, color='b', label="Training")
        plt.plot(val_losses, color='r', label="Validation")
        plt.legend()

        # Assign plot name conditionally depending on model type
        title = self.model.__class__.__name__
        if title == "RNNLanguageModel":
            title = "Vanilla RNN"
        elif title == "LSTMLanguageModel":
            title = "LSTM"
        elif title == "TransformerLanguageModel":
            title = "Transformer"
        else:
            title = "Model"

        plt.title(f"{title} Loss Curves", size=16)
        plt.xlabel("Epochs", size=16)
        plt.ylabel("Cross Entropy", size=14)

        # Save plot if file path was provided
        if file_path is not None:
            plt.savefig(file_path)
        plt.show()
