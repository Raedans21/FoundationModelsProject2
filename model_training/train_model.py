import torch
from torch.utils.data import DataLoader
from FoundationModelsProject2.data_manipulation import TextDataset, collate_fn
from FoundationModelsProject2.tokenizer import load_tokenizer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(device, model, tokenizer_path, train_file_path, val_file_path, max_seq_len, batch_size, learning_rate, epochs, no_improve_threshold=5):
    """
    Train and validate the given model
    :param device: Torch device to run training on
    :param model: NN subclass model instance
    :param tokenizer_path: file path to trained tokenizer
    :param train_file_path: file path to training dataset
    :param val_file_path: file path to validation dataset
    :param max_seq_len: maximum sequence length allowed to be passed to the model
    :param batch_size: size of each batch
    :param learning_rate: model training learning rate
    :param epochs: number of epochs to train the model for
    :param no_improve_threshold: number of epochs with no improvement to wait before early stopping
    :return: trained model, and training and validation losses for each epoch
    """
    # Utilize the best available device and move model to it
    model.to(device)

    # load in tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_piece_size()

    # Load in the training and validation datasets
    train_dataset = TextDataset(train_file_path, tokenizer, max_seq_len)
    val_dataset = TextDataset(val_file_path, tokenizer, max_seq_len)

    # This will handle batching and shuffling during training. collate_fn handles padding of uneven sequences
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Using AdamW optimizer on the trainable params. This is a standard for LMs
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # Using a learning rate scheduler that reduces LR by half after stagnation for 1 epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=3)

    best_val_loss = float('inf') # keep track of the best validation loss
    no_improve_epochs = 0

    # Store train and validation loss curves
    train_losses, val_losses, = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        # Loop through each sample batch in training
        for input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move input and target tensors to device memory
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            # Reset gradients between batches
            optimizer.zero_grad()
            # Compute output logits
            train_outputs = model(input_ids)
            # Handle different model output formats
            if isinstance(train_outputs, tuple):
                logits, _ = train_outputs
            else:
                logits = train_outputs
            # Apply cross entropy and backpropagation
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation processing
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in tqdm(val_loader, desc=f"Epoch {epoch+1}"):
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                val_outputs = model(input_ids)
                # Handle different model output formats
                if isinstance(val_outputs, tuple):
                    logits, _ = val_outputs
                else:
                    logits = val_outputs
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Handle non-improvement epochs and early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            scheduler.step(val_losses[-1])

        # Exit training if non-improvement epochs exceeds non-improvement epoch threshold
        if no_improve_epochs > no_improve_threshold:
            break

    print(f"Final Training Loss: {train_losses[-1]:.3f}")
    print(f"Final Validation Loss: {val_losses[-1]:.3f}")
    return train_losses, val_losses, model