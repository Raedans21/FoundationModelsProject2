import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=6, dropout=0.2, pad_token_id=0):
        """
        Create an LSTM Language Model.
        :param vocab_size: size of the vocabulary
        :param embed_dim: size of each token's embedding vector
        :param hidden_dim: size of the LSTM hidden states
        :param num_layers: number of LSTM layers to stack
        :param dropout: training dropout rate
        :param pad_token_id: token ID of <pad> token
        """
        super(LSTMLanguageModel, self).__init__()
        # Define embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        # Define stacked LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        # Output layer that maps hidden state of final LSTM to output
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        """
        Compute model output logits given a sequence.
        :param input_ids: Sequence of input token IDs, Tensor of shape (batch_size, seq_len)
        :param hidden: previous hidden state, a tuple composed of (hidden, cell) tensors
        :return: output logits, Tensor of shape (batch_size, seq_len, vocab_size)
        """
        embeds = self.embedding(input_ids) # compute embeddings for all input tokens in parallel
        output, hidden = self.lstm(embeds, hidden) # pass embeddings through the LSTM layers
        logits = self.fc(output) # compute output logits
        return logits, hidden

    def predict_next_token(self, input_ids, temperature=1.0):
        """
        Predict the next token ID (and hidden state) from the last token in input_ids using top p sampling.
        :param input_ids: Input sequence token IDs
        :param temperature: setting for sampling
        :return: next token ID, hidden state
        """
        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature

            # Sort output logits to apply top-p filtering.
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            # Compute the probability distribution of the sorted logits.
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            # Compute the cumulative sum of the sorted logit probabilities.
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Nucleus sampling p value
            top_p = 0.9
            # Create boolean mask to remove cumulative probabilities greater than top_p
            mask = cumulative_probs > top_p
            # Avoid masking the first token to handle situations when the first token probability is greater than top_p
            mask[..., 0] = False
            # Set any masked elements to negative infinity to set probability to zero in softmax call
            sorted_logits[mask] = -float('Inf')
            # Transform the filtered logits back to probabilities, zeroing out masked token_ids
            filtered_probs = torch.softmax(sorted_logits, dim=-1)
            # Sample one token index from the filtered probability distribution using weighted sampling
            next_token_index = torch.multinomial(filtered_probs, num_samples=1)
            # Map the selected token index back to onto the original vocabulary indices
            next_token_id = sorted_indices.gather(1, next_token_index)
            return next_token_id.item(), hidden

    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, device='cpu'):
        """
        Generate a full output sequence given a prompt.

        :param tokenizer: The trained SentencePiece tokenizer
        :param prompt: The input prompt (plain text string)
        :param max_length: Maximum number of tokens to generate auto-regressively before stopping
        :param eos_token_id: The token ID of the EOS token
        :param temperature: Temperature setting for sampling
        :param device: Device we are using to run the model
        """
        self.eval()
        input_ids = tokenizer.encode(prompt, out_type=int) # Encode the input string into token IDs
        # convert token ID list to a tensor, move to device memory, and add a batch dimension (1D to 2D)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = []
        hidden = None # initial hidden state is None

        # loop over map output tokens
        for _ in range(max_length):
            next_token_id, hidden = self.predict_next_token(input_tensor, temperature)
            # exit early if the model generated <eos> token ID
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            # keep track of generated tokens
            generated_ids.append(next_token_id)
            # the input to the next step is just the new token and the hidden state
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        # decode generated token IDs into tokens
        return tokenizer.decode(generated_ids, out_type=str)